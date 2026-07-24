# Needle decode benchmark: stock / KV / JumpForward / KV+JumpForward

40 prompts (all 10 tools shown each call), greedy, CPU JAX. Prefill = encoder ingest (+cross-KV projection for KV configs). Generation = output-token decode loop.

| config | prefill tok/s | **gen tok/s** | mean fwd passes | mean out tok | mean total ms | end-to-end speedup |
|---|---|---|---|---|---|---|
| stock (full-buffer) | 1,503 | **17** | 20.7 | 19.7 | 1558 | 1.0x |
| KV-cache | 886 | **101** | 20.7 | 19.7 | 866 | 1.8x |
| JumpForward | 1,503 | **37** | 11.6 | 20.6 | 958 | 1.6x |
| KV + JumpForward | 886 | **221** | 13.6 | 20.6 | 765 | 2.0x |

## Reading the axes

- **KV-cache win** (same unconstrained output): stock -> KV = 6.0x gen tok/s.
- **Jump-forward win** (same full-buffer path): stock -> JF = 2.2x gen tok/s (fewer model passes on the same output).
- **Combined**: stock -> KV+JF = 13.0x gen tok/s.
- KV-cache under jump-forward: JF -> KV+JF = 6.0x.
- Jump-forward under KV-cache: KV -> KV+JF = 2.2x.
- The two wins are ~orthogonal and multiply: 6.0x x 2.2x ~= 13.0x.

**Prefill vs generation tradeoff.** KV's prefill tok/s is *lower* (it projects cross-K/V once, up front) but that is exactly the per-token work stock repeats every step — so end-to-end latency still drops sharply (see 'mean total ms'). The cross-KV front-load pays for itself within the first couple of output tokens.

Notes: mean-fwd-passes for stock/KV = output tokens + 1 (one pass per token); for JF/KV+JF it is the number of *model* passes after collapsing grammar-forced runs into batched extends. Absolute tok/s are CPU-bound and un-cached per-pass; the *ratios* are the result. On the un-cached decode each stock pass re-runs the whole decoder over the full buffer AND re-projects cross-K/V over all encoder positions — that reprojection is exactly what the KV config caches once at prefill.
