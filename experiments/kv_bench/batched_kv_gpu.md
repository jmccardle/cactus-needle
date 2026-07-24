# Needle batched KV-cache decode (GPU / NVIDIA GeForce RTX 4090)

Real greedy generation (EOS-stopping), batch 1..256, max_new=48, all 10 tools shown each row. Aggregate = total generated tokens / gen seconds.

**Parity** (20 prompts spanning all 10 domains). Batch-invariance (a row decoded in the full batch vs decoded alone): **20/20** — exact; the cache math carries the batch axis with no error. Agreement with single-stream stock (bf16 flax) greedy: **20/20** (any gap is the same fp32-vs-bf16 near-tie as `kv_model`'s 49/50 CPU parity).

| batch | agg gen tok/s | per-stream tok/s | mean out tok | peak GPU MB | agg scaling | per-stream retention |
|---|---|---|---|---|---|---|
| 1 | 1,131 | 1,131 | 27.0 | 891 | 1.0x | 100% |
| 4 | 3,708 | 927 | 30.5 | 891 | 3.3x | 82% |
| 16 | 11,723 | 733 | 31.3 | 891 | 10.4x | 65% |
| 64 | 18,100 | 283 | 23.2 | 2,343 | 16.0x | 25% |
| 128 | 17,972 | 140 | 20.6 | 4,475 | 15.9x | 12% |
| 256 | 21,511 | 84 | 21.6 | 8,709 | 19.0x | 7% |

## Reading it

- **Aggregate throughput scales to ~21,511 tok/s** at batch 256 (19x over batch 1) and keeps climbing far past where the un-cached full-buffer path saturated (~9,200 tok/s at B=16, `batched_throughput.py`).
- **Per-stream throughput is retained** much better than the full-buffer path (which fell to ~2% by B=256): here each decode step is O(1) in sequence length, not O(max_dec), so batching is close to free until the batch actually fills the card's compute.
- Peak GPU memory at B=256 is 8,709 MB (cross-K/V cached as KV-heads keeps it well under 24 GB), so larger batches are available if throughput has not yet saturated.

**This is the serving path.** Batched cached decode + an llguidance grammar (one union grammar, applied per row) + a fine-tuned checkpoint = many requests in, one guaranteed-valid tool call each. The batch axis here and the grammar/jump-forward axis in `../llguidance_poc` are orthogonal; a batched grammar-masked decode combines them. Numbers are greedy/argmax; absolute tok/s are backend-specific, the scaling is the point.
