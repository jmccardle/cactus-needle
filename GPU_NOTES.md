# Running Needle on CUDA — setup + what actually changes

The model is pure device-agnostic Flax; getting it on a GPU is a packaging change, not a
code change. But the switch surfaced two benchmark bugs (invisible on CPU) and inverts one
of the headline conclusions from `experiments/kv_bench`. Both are documented here.

Verified on: **RTX 4090 (24 GB, Ada, compute 8.9), driver 570.86, JAX 0.10.2**.

## 1. Install (the only required step)

CPU-only `jaxlib` was installed. Add the CUDA plugin, **version-locked to jax**:

```bash
/storage/needle-e1/repo/.venv/bin/pip install "jax[cuda12]==0.10.2" "jaxlib==0.10.2"
```

This pulls `jax-cuda12-plugin`, `-pjrt`, and the bundled `nvidia-*-cu12` runtime wheels
(cuDNN 9, cuBLAS, cuSPARSE, …). No system CUDA toolkit is needed — driver 570 runs the
CUDA 12.x runtime that ships inside the wheels. The repo's `pyproject.toml` already declares
`gpu = ["jax[cuda12]"]`; pin the version when you use it, or a plugin/jaxlib mismatch will
silently fall back to CPU.

## 2. The loader gotcha (cost us the first launch)

The box has a **system CUDA 12.8** on `LD_LIBRARY_PATH` (`/usr/local/cuda-12.8/lib64`).
jaxlib 0.10.2's init dlopens `libcusparse.so.12`, finds the *system* one, and its version
check rejects it:

```
RuntimeError: Unable to load cuSPARSE. Is it installed?  (cusparseGetProperty ... failed)
```

Fix: put the pip wheel libs first on the loader path so jaxlib uses its own matched runtime.

```bash
SP=/storage/needle-e1/repo/.venv/lib/python3.11/site-packages
export LD_LIBRARY_PATH="$(ls -d $SP/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cuda
```

With that, `jax.devices()` → `[CudaDevice(id=0)]`. The `JAX_PLATFORMS=cuda` env var overrides
the `os.environ.setdefault("JAX_PLATFORMS","cpu")` pins in the experiment scripts without
editing them. (A permanent fix is to drop the system-CUDA entry from `LD_LIBRARY_PATH`, or
prepend the wheel dirs in the venv activate script.)

## 3. Run

```bash
cd experiments/kv_bench
python bench.py                 # 4-config batch-1 A/B -> results_gpu.md (backend-suffixed)
python batched_throughput.py    # batch 1..256 throughput sweep -> batched_throughput_gpu.md
python parity.py                # KV vs stock greedy: 50/50 exact on GPU
```

Reports are backend-aware and write to `results.md` on CPU / `results_gpu.md` on GPU, so the
two never clobber each other.

## 4. What the GPU changes (the honest part)

**Correctness is fine.** KV-vs-stock greedy parity is **50/50** on the 4090 (the one CPU
near-tie flipped clean — bf16 accumulation differs slightly on-device, as expected, but not
in a way that hurts).

**A full-buffer decode pass is ~0.6 ms on GPU** (measured). The 26M model is far too small to
occupy a 4090 one stream at a time, so at **batch 1 the wall time is host-loop / dispatch
bound, not model-compute bound.** That inverts the CPU conclusions:

| gen tok/s (batch 1) | stock | KV | JumpForward | KV+JF | KV win | JF win | combined |
|---|---|---|---|---|---|---|---|
| **CPU** (jax cpu) | 19 | 123 | 33 | 196 | 6.6× | 1.8× | **10.5×** (compounds) |
| **GPU** (4090) | 1,026 | 1,117 | 1,663 | 1,223 | 1.1× | 1.6× | **1.2×** (JF alone wins) |

- On **CPU** each pass is expensive compute, so KV-cache (cheaper passes) and jump-forward
  (fewer passes) attack different costs and **multiply** (6.6× × 1.8× ≈ 10.5×).
- On **GPU batch-1** the pass they optimize is already ~free. KV-cache adds per-pass kernel/
  host overhead (dynamic-slice cache writes, more small ops) and **KV+JumpForward (1,243) is
  actually slower than JumpForward alone (1,684)**. Jump-forward still helps (1.6×) — but only
  because it runs *fewer host iterations*, not fewer FLOPs. End-to-end latency is flat across
  all four configs (~380 ms).

**The GPU lever is batching, not the decode tricks.** `batched_throughput.py` (stock
full-buffer decode, parallelised across the batch, on-device argmax):

| batch | agg gen tok/s | per-stream | step ms | peak GPU MB | scaling |
|---|---|---|---|---|---|
| 1 | 1,475 | 1,475 | 0.7 | 147 | 1.0× |
| 4 | 4,197 | 1,049 | 1.0 | 229 | 2.8× |
| 16 | **9,191** | 574 | 1.7 | 583 | **6.2×** |
| 64 | 8,956 | 140 | 7.1 | 2,004 | 6.1× |
| 256 | 8,341 | 33 | 30.7 | 7,760 | 5.7× |

Aggregate throughput climbs ~6× to batch 16, then **saturates and declines** — because this
is the *un-cached* full-buffer path (each step recomputes the whole 64-slot buffer + re-projects
cross-K/V), so it is FLOP-bound and batching only fills the idle compute. Per-stream tok/s
falls the whole way. Params are ~52 MB bf16; even batch 256 peaks at 7.8 GB, far under 24 GB —
the ceiling here is compute, not memory.

### The batched KV-cache decode (`kv_batch.py`) — the real win, now built

`BatchKVNeedle` carries a batch axis through the encoder, cross-K/V projection, self-attn cache,
and chunked `extend`. Real greedy generation, `batched_kv_bench.py` (batch-invariance **20/20**,
stock agreement **20/20**):

| batch | agg gen tok/s | per-stream tok/s | peak GPU MB | agg scaling |
|---|---|---|---|---|
| 1 | 1,108 | 1,108 | 891 | 1.0× |
| 16 | 11,137 | 696 | 891 | 10.1× |
| 64 | 18,088 | 283 | 2,343 | 16.3× |
| 256 | **21,635** | 85 | 8,709 | **19.5×** |

Because each cached decode step is **O(1) in sequence length** (not O(max_dec) like the
full-buffer path), it does **not** saturate at B=16: aggregate throughput keeps climbing to
**~21,600 tok/s at batch 256 — 2.4× past the full-buffer path's ~9,200 tok/s ceiling** — and
per-stream throughput holds far better. Batch and KV-cache are orthogonal and compound, exactly
as predicted. Cross-K/V are cached as KV-heads (repeated to H in-kernel) so B=256 still fits in
8.7 GB of the 24. Fine-tuned + grammar-constrained, this is the serving path for Needle as a
batched classifier / structured-extraction engine.

## 5. Two benchmark bugs the GPU surfaced (now fixed)

Both were latent on CPU and only bit on-device; fixing them also corrected the *CPU* headline
(the earlier committed `results.md` reported ~13× combined — that was partly artifact; the
honest CPU figure is 10.5×).

1. **Per-prompt XLA recompilation.** `encode()` returned a *variable-length* encoder output,
   so the full-buffer `_decode` re-JIT'd for every new encoder length — a ~1.8 s/prompt tax
   on GPU that landed entirely on whichever decode arm ran first (stock), making its tok/s
   ~30× too low and every "speedup" over it fictitious. Fix: pad the encoder input to a fixed
   `ENC_PAD`/`enc_max` (640); pad positions are masked, so output is unchanged (parity 10/10).
   (`needle_llg.py`, `kv_model.py`.)

2. **Async-dispatch timing.** JAX dispatches async. The jump-forward arms inject grammar-forced
   token runs *without reading logits*, so they queued `_decode` calls the timer never waited
   on — inflating their tok/s. Fix: `block_until_ready` on the final logits before stopping the
   timer in `gen_constrained` and `kv_jf`. (No-op on CPU; correct on GPU.)

## 6. Training / fine-tuning

Same install. `needle/utils/distributed.py` already builds a `jax.local_devices()` mesh, so
training uses the 4090 (or multiple GPUs) with no code change. Params + activations for a 26M
model are tiny relative to 24 GB — the constraint will be batch/throughput, not capacity.
