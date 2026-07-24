# Needle decode benchmark: stock / KV / JumpForward / KV+JumpForward

40 prompts (all 10 tools shown each call), greedy, CPU JAX (cpu). Prefill = encoder ingest (+cross-KV projection for KV configs). Generation = output-token decode loop.

| config | prefill tok/s | **gen tok/s** | mean fwd passes | mean out tok | mean total ms | end-to-end speedup |
|---|---|---|---|---|---|---|
| stock (full-buffer) | 3,504 | **19** | 20.7 | 19.7 | 1229 | 1.0x |
| KV-cache | 1,881 | **123** | 20.7 | 19.7 | 477 | 2.6x |
| JumpForward | 3,504 | **33** | 11.6 | 20.6 | 786 | 1.6x |
| KV + JumpForward | 1,881 | **196** | 13.6 | 20.6 | 422 | 2.9x |

## Reading the axes

- **KV-cache win** (same unconstrained output): stock -> KV = 6.6x gen tok/s.
- **Jump-forward win** (same full-buffer path): stock -> JF = 1.8x gen tok/s (fewer model passes on the same output).
- **Combined**: stock -> KV+JF = 10.5x gen tok/s.
- KV-cache under jump-forward: JF -> KV+JF = 5.9x.
- Jump-forward under KV-cache: KV -> KV+JF = 1.6x.
- The two wins are ~orthogonal and multiply: 6.6x x 1.8x ~= 10.5x. Each attacks a different cost (KV = per-pass work, jump-forward = number of passes), so on this backend they compound.

**Prefill vs generation tradeoff.** KV's prefill tok/s is *lower* (it projects cross-K/V once, up front) but that is exactly the per-token work stock repeats every step — so end-to-end latency drops (1229 -> 477 ms; see 'mean total ms'). The cross-KV front-load pays for itself within the first couple of output tokens.

Notes: mean-fwd-passes for stock/KV = output tokens + 1 (one pass per token); for JF/KV+JF it is the number of *model* passes after collapsing grammar-forced runs into batched extends. Absolute tok/s are CPU-bound and un-cached per-pass; the *ratios* are the result. On the un-cached decode each stock pass re-runs the whole decoder over the full buffer AND re-projects cross-K/V over all encoder positions — that reprojection is exactly what the KV config caches once at prefill.
