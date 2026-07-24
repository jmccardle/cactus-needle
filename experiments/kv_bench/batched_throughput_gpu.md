# Needle batched decode throughput (GPU / NVIDIA GeForce RTX 4090)

Stock full-buffer decode (needle/model/run.py step), parallelised across the batch, on-device argmax, 40 fixed decode steps per config, max_gen=64. Aggregate = batch x steps / gen seconds.

| batch | agg gen tok/s | per-stream tok/s | step ms | prefill tok/s | peak GPU MB | scaling vs B=1 |
|---|---|---|---|---|---|---|
| 1 | 1,475 | 1,475 | 0.7 | 169 | 147 | 1.0x |
| 4 | 4,197 | 1,049 | 1.0 | 728 | 229 | 2.8x |
| 16 | 9,191 | 574 | 1.7 | 3,160 | 583 | 6.2x |
| 64 | 8,956 | 140 | 7.1 | 12,249 | 2,004 | 6.1x |
| 256 | 8,341 | 33 | 30.7 | 38,911 | 7,760 | 5.7x |

## Reading it

- **At batch 1 the 26M model barely wakes the card**: 0.7 ms/step, 147 MB, 1,475 tok/s. (This is already with on-device argmax — the batch-1 A/B in `bench.py`, which syncs a Python `int(argmax)` to the host every token, is much slower again. Removing that per-token host sync is the first and cheapest GPU win.)
- **Batching recovers ~6x aggregate throughput, then saturates** at batch 16 (9,191 tok/s) and *declines* beyond it (8,341 tok/s at B=256). Per-step time stays ~flat while the batch is small (0.7->1.7 ms up to B=16) then grows linearly (30.7 ms at B=256).
- **Per-stream tok/s falls steadily** (1,475 -> 33) — the opposite of a bandwidth-bound model with free parallelism. That's because this is the **un-cached full-buffer** decode: each step recomputes the entire 64-slot buffer and re-projects cross-K/V, so it is **FLOP-bound**. Batching fills the idle compute (the 1->16 win), but past saturation you are just paying for redundant work.
- Peak GPU memory stays modest (7,760 MB at B=256; params are ~52 MB bf16), so the ceiling here is compute, not the 24 GB.

**The real lever is the KV cache, not batch.** This path does ~64x redundant work per token by design. A batched KV-cache decode (`kv_model.py`, currently batch-1) would cut per-step FLOPs by roughly that factor and push the saturation point far higher — batch and KV-cache are orthogonal and compound. Constrained/jump-forward decoding (host-side grammar) does not batch trivially and is excluded here to isolate raw decode throughput.
