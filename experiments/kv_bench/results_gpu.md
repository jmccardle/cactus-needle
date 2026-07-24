# Needle decode benchmark: stock / KV / JumpForward / KV+JumpForward

40 prompts (all 10 tools shown each call), greedy, GPU JAX (NVIDIA GeForce RTX 4090). Prefill = encoder ingest (+cross-KV projection for KV configs). Generation = output-token decode loop.

| config | prefill tok/s | **gen tok/s** | mean fwd passes | mean out tok | mean total ms | end-to-end speedup |
|---|---|---|---|---|---|---|
| stock (full-buffer) | 1,624 | **1,026** | 20.7 | 19.7 | 386 | 1.0x |
| KV-cache | 1,613 | **1,117** | 20.7 | 19.7 | 387 | 1.0x |
| JumpForward | 1,624 | **1,663** | 11.6 | 20.6 | 379 | 1.0x |
| KV + JumpForward | 1,613 | **1,223** | 13.6 | 20.6 | 386 | 1.0x |

## Reading the axes

- **KV-cache win** (same unconstrained output): stock -> KV = 1.1x gen tok/s.
- **Jump-forward win** (same full-buffer path): stock -> JF = 1.6x gen tok/s (fewer model passes on the same output).
- **Combined**: stock -> KV+JF = 1.2x gen tok/s.
- KV-cache under jump-forward: JF -> KV+JF = 0.7x.
- Jump-forward under KV-cache: KV -> KV+JF = 1.1x.
- **Combining is not additive here — the best single config wins.** stock -> KV = 1.1x, stock -> JF = 1.6x, but stock -> KV+JF is only 1.2x, *below* JF alone (JF -> KV+JF = 0.7x). On this backend a full-buffer pass is already compute-negligible, so the KV-cache optimizes work that is already free while adding per-pass kernel/host overhead (dynamic-slice cache writes, more small ops, grammar consume). The bottleneck has shifted from per-pass FLOPs (compute-bound, CPU) to per-pass dispatch/host latency (launch-bound, GPU batch-1) — where KV-cache and jump-forward stop compounding and batching (see batched_throughput) is the lever instead.

**Prefill vs generation tradeoff.** KV's prefill tok/s is *lower* (it projects cross-K/V once, up front). On this backend that front-load does **not** pay off: end-to-end latency is flat across configs (386 vs 387 ms) because the un-cached full-buffer pass it eliminates is already ~free here — the wall time is host-loop/dispatch bound, not model-compute bound.

Notes: mean-fwd-passes for stock/KV = output tokens + 1 (one pass per token); for JF/KV+JF it is the number of *model* passes after collapsing grammar-forced runs into batched extends. Absolute tok/s are GPU-bound and un-cached per-pass; the *ratios* are the result. On the un-cached decode each stock pass re-runs the whole decoder over the full buffer AND re-projects cross-K/V over all encoder positions — that reprojection is exactly what the KV config caches once at prefill.
