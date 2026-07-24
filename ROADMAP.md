# cactus-needle — fork roadmap (jmccardle)

Downstream fork of [cactus-compute/needle](https://github.com/cactus-compute/needle). This file
tracks **base-model and base-software** work done in *this* repo. JMFTS-specific fine-tuning and
the shipped specialized component live in the JMFTS repo (`~/Development/jmfts`: `ROADMAP.md` →
"Open Research Questions" + the `jmfts-needle/` component), **not here**.

## Why this fork exists

JMFTS (a Postgres/pgvector retrieval + agent-memory substrate) is adopting needle as a ~0-cost
NL→tool-call router for content-driven ingestion and query routing. The load-bearing downstream
idea — **ToolRAG: use JMFTS's own pgvector to retrieve the top-k relevant tools per call** so the
26M model only ever sees ~3–6 candidates — shapes what the base model needs to be good at: load
safely and fast, and ideally expose a tool-ranking signal and calibrated abstention.

## Near-term

### 1. Stop shipping as a pickle (upstream #36) — SCHEDULED
Migrate model serialization from pickle (the `.pkl` JAX/Flax checkpoint) to **safetensors** (raw
tensor buffers + JSON metadata). Rationale per upstream #36: **security** (Sleepy-Pickle-class
arbitrary-code-execution on load), **brittleness** (pickle stores object definitions, breaks on
refactor/dep bumps), **performance** (unpickling forces allocs/CPU vs. zero-copy load).
- [ ] Audit every save/load site: `needle/model/export.py`, `needle/model/run.py`,
  `needle/training/{pretrain,finetune,train,eval}.py`, `needle/cli.py`, `needle/ui/server.py`,
  `needle/utils/distributed.py`.
- [ ] Flax params are a **nested pytree** — implement a deterministic flatten → `{dotted.key: array}`
  for `safetensors.flax.save_file`, with structural metadata in the JSON header and an unflatten on
  load. (safetensors is flat-tensors-only; the tree structure must be recorded separately.)
- [ ] One-shot `.pkl → .safetensors` converter for the released checkpoint.
- [ ] Legacy load shim behind an explicit opt-in flag so existing `.pkl` checkpoints aren't orphaned;
  default to safetensors.
- [ ] Parity gate: identical logits pre/post conversion on a fixed eval batch.
- Reference (don't depend): a community port `Abdalrahman/needle-rs-safetensors` already exists.

### 2. Verify + document the real context window
The released `config.json` has no `max_seq_len` / `max_position_embeddings`; the dataclass default
(128) is the *toy* config, not the 26M model (d=512, 12enc/8dec, RoPE θ=10000). Empirically pin the
usable encoder budget and document it — **both** JMFTS's size-gate recursion and the ToolRAG top-k
budget depend on this exact number.

### 3. Tool-ranking signal + decoder logprobs for ToolRAG/confidence (investigate)
- Does the model already carry a usable tool/name embedding or contrastive head that could rank
  tools (so JMFTS could reuse it) — or is JMFTS better off embedding tool descriptions independently
  in pgvector? Scope both.
- Are decoder logprobs on the tool-name tokens accessible? JMFTS's confidence bands depend on it;
  absent that, JMFTS falls back to self-consistency sampling (more expensive). Document the answer.

### 4. First-class abstention / no-op (investigate, likely downstream)
The model over-calls (no trained abstention; "given any input it emits some call"). Evaluate whether
a *base-model* change helps — a reserved no-op token (à la Octopus functional tokens) or training-data
changes — vs. handling it purely downstream (JMFTS adds an explicit `no_applicable_operation` tool +
fine-tunes with hard negatives). Recommendation leans downstream, but scope the base-model option.

## Later / speculative
- ONNX / GGUF export paths for deployment outside the Cactus runtime (JMFTS may run it under its own
  stack on the GPU server / CPU).
- Batch-inference throughput characterization for corpus-wide sweeps (the "affordance sweeper" use).

## Boundary — what does NOT belong in this repo
- The JMFTS op-catalog and its tool schemas → JMFTS (`@expose` registry emits them).
- Fine-tuning data (JSONL `{query, tools, answers}` + hard negatives) and the fine-tuned weights →
  `jmfts-needle/` in the JMFTS repo.
- The ToolRAG op-embedding index (built from JMFTS's own embeddings) → JMFTS repo.
