"""Batched full-buffer decode throughput: where the GPU actually earns its keep.

The batch-1 A/B in bench.py is latency/host-sync bound — the 26M model is far too
small to saturate a 4090 one stream at a time. This sweep runs the *stock* un-cached
full-buffer decode across a batch of independent prompts, with a single on-device
argmax per step (no per-row host sync), for a FIXED number of steps. Aggregate
generation throughput = (batch * steps) / gen_seconds.

It measures the repo's existing decode path (needle/model/run.py full-buffer step),
just parallelised across the batch — so the number is honest about what today's code
does, and shows the throughput headroom the tiny model leaves on the table at batch 1.
"""
import os, sys, json, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llguidance_poc"))
import numpy as np
import jax, jax.numpy as jnp

from needle.model.run import load_checkpoint, _build_encoder_input, _get_decode_fn
from needle.model.architecture import SimpleAttentionNetwork, make_padding_mask
from needle.dataset.dataset import get_tokenizer, DEFAULT_MAX_ENC_LEN
from tools_spec import tools_json_all

CKPT = "/storage/needle-e1/weights/needle.pkl"
DATA = os.path.join(os.path.dirname(__file__), "..", "llguidance_poc", "data")
OUT = os.path.dirname(__file__)
MAX_GEN = 64
STEPS = 40                      # fixed decode steps timed per config
BATCHES = [1, 4, 16, 64, 256]


def load_prompts():
    qs = []
    for f in sorted(glob.glob(os.path.join(DATA, "*.json"))):
        for ex in json.load(open(f)):
            qs.append(ex["query"])
    return qs


def batched_encode(model, params, tok, queries, tools):
    pad_id = tok.pad_token_id
    lists = [_build_encoder_input(tok, q, tools, DEFAULT_MAX_ENC_LEN) for q in queries]
    max_enc = max(len(t) for t in lists)
    arr = np.full((len(lists), max_enc), pad_id, dtype=np.int32)
    for i, t in enumerate(lists):
        arr[i, :len(t)] = t
    enc_input = jnp.asarray(arr)
    src_mask = make_padding_mask(enc_input, pad_id)
    enc_out, enc_mask = model.apply({"params": params}, enc_input,
                                    src_mask=src_mask, method="encode")
    return enc_out, enc_mask, max_enc


def make_step(model, params, max_gen, eos_id, pad_id):
    """Jitted: one batched full-buffer forward + on-device argmax write at `pos`."""
    decode_fn = _get_decode_fn(model, max_gen)

    @jax.jit
    def step(dec_buffer, encoder_out, enc_mask, pos):
        logits = decode_fn(params, dec_buffer, encoder_out, enc_mask)  # (B, max_gen, V)
        nxt = jnp.argmax(logits[:, pos], axis=-1).astype(jnp.int32)    # (B,)
        dec_buffer = jax.lax.dynamic_update_slice(
            dec_buffer, nxt[:, None], (0, pos + 1))
        return dec_buffer
    return step


def run_batch(model, params, tok, step, enc_out, enc_mask, B, steps):
    pad_id, eos_id = tok.pad_token_id, tok.eos_token_id
    dec = np.full((B, MAX_GEN), pad_id, dtype=np.int32)
    dec[:, 0] = eos_id
    dec = jnp.asarray(dec)
    # warm (compile this B-shape) then time
    dec = step(dec, enc_out, enc_mask, jnp.int32(0)); dec.block_until_ready()
    dec = np.full((B, MAX_GEN), pad_id, dtype=np.int32); dec[:, 0] = eos_id
    dec = jnp.asarray(dec)
    t0 = time.perf_counter()
    for pos in range(steps):
        dec = step(dec, enc_out, enc_mask, jnp.int32(pos))
    dec.block_until_ready()
    return time.perf_counter() - t0


def gpu_mem_mb():
    try:
        s = jax.devices()[0].memory_stats()
        return s.get("peak_bytes_in_use", s.get("bytes_in_use", 0)) / 1e6
    except Exception:
        return float("nan")


def main():
    backend = jax.default_backend()
    dev = jax.devices()[0]
    devname = getattr(dev, "device_kind", str(dev))
    params, cfg = load_checkpoint(CKPT)
    model = SimpleAttentionNetwork(cfg)
    tok = get_tokenizer()
    tools = tools_json_all()
    prompts = load_prompts()
    step = make_step(model, params, MAX_GEN, tok.eos_token_id, tok.pad_token_id)

    rows = []
    for B in BATCHES:
        qs = [prompts[i % len(prompts)] for i in range(B)]
        t = time.perf_counter()
        enc_out, enc_mask, max_enc = batched_encode(model, params, tok, qs, tools)
        jax.block_until_ready(enc_out)
        prefill_s = time.perf_counter() - t
        gen_s = run_batch(model, params, tok, step, enc_out, enc_mask, B, STEPS)
        gen_toks = B * STEPS
        rows.append({
            "B": B, "prefill_s": prefill_s, "prefill_toks": B * max_enc,
            "gen_s": gen_s, "gen_toks": gen_toks,
            "agg_tps": gen_toks / gen_s, "per_stream_tps": (gen_toks / gen_s) / B,
            "step_ms": 1000 * gen_s / STEPS, "mem_mb": gpu_mem_mb(),
        })
        print(f"  B={B:4d}  agg {rows[-1]['agg_tps']:9,.0f} tok/s  "
              f"per-stream {rows[-1]['per_stream_tps']:7,.0f}  "
              f"step {rows[-1]['step_ms']:6.1f} ms  mem {rows[-1]['mem_mb']:6.0f} MB",
              flush=True)

    write_report(rows, backend, devname)


def write_report(rows, backend, devname):
    b1 = rows[0]["agg_tps"]
    lines = ["# Needle batched decode throughput ({} / {})\n".format(backend.upper(), devname)]
    lines.append(f"Stock full-buffer decode (needle/model/run.py step), parallelised across the "
                 f"batch, on-device argmax, {STEPS} fixed decode steps per config, max_gen={MAX_GEN}. "
                 f"Aggregate = batch x steps / gen seconds.\n")
    lines.append("| batch | agg gen tok/s | per-stream tok/s | step ms | prefill tok/s | "
                 "peak GPU MB | scaling vs B=1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        pf = r["prefill_toks"] / r["prefill_s"]
        lines.append(f"| {r['B']} | {r['agg_tps']:,.0f} | {r['per_stream_tps']:,.0f} | "
                     f"{r['step_ms']:.1f} | {pf:,.0f} | {r['mem_mb']:,.0f} | "
                     f"{r['agg_tps'] / b1:.1f}x |")
    peak = max(rows, key=lambda r: r["agg_tps"])
    step1 = rows[0]["step_ms"]
    lines.append("\n## Reading it\n")
    lines.append(f"- **At batch 1 the 26M model barely wakes the card**: {step1:.1f} ms/step, "
                 f"{rows[0]['mem_mb']:.0f} MB, {b1:,.0f} tok/s. (This is already with on-device "
                 f"argmax — the batch-1 A/B in `bench.py`, which syncs a Python `int(argmax)` to the "
                 f"host every token, is much slower again. Removing that per-token host sync is the "
                 f"first and cheapest GPU win.)")
    lines.append(f"- **Batching recovers ~{peak['agg_tps'] / b1:.0f}x aggregate throughput, then "
                 f"saturates** at batch {peak['B']} ({peak['agg_tps']:,.0f} tok/s) and *declines* "
                 f"beyond it ({rows[-1]['agg_tps']:,.0f} tok/s at B={rows[-1]['B']}). Per-step time "
                 f"stays ~flat while the batch is small ({step1:.1f}->{rows[2]['step_ms']:.1f} ms up "
                 f"to B={rows[2]['B']}) then grows linearly ({rows[-1]['step_ms']:.1f} ms at "
                 f"B={rows[-1]['B']}).")
    lines.append(f"- **Per-stream tok/s falls steadily** ({rows[0]['per_stream_tps']:,.0f} -> "
                 f"{rows[-1]['per_stream_tps']:,.0f}) — the opposite of a bandwidth-bound model with "
                 f"free parallelism. That's because this is the **un-cached full-buffer** decode: each "
                 f"step recomputes the entire {MAX_GEN}-slot buffer and re-projects cross-K/V, so it is "
                 f"**FLOP-bound**. Batching fills the idle compute (the 1->16 win), but past saturation "
                 f"you are just paying for redundant work.")
    lines.append(f"- Peak GPU memory stays modest ({rows[-1]['mem_mb']:,.0f} MB at B={rows[-1]['B']}; "
                 f"params are ~52 MB bf16), so the ceiling here is compute, not the 24 GB.")
    lines.append("\n**The real lever is the KV cache, not batch.** This path does "
                 f"~{MAX_GEN}x redundant work per token by design. A batched KV-cache decode "
                 "(`kv_model.py`, currently batch-1) would cut per-step FLOPs by roughly that factor "
                 "and push the saturation point far higher — batch and KV-cache are orthogonal and "
                 "compound. Constrained/jump-forward decoding (host-side grammar) does not batch "
                 "trivially and is excluded here to isolate raw decode throughput.")
    suffix = "" if backend == "cpu" else "_" + backend
    open(os.path.join(OUT, f"batched_throughput{suffix}.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
