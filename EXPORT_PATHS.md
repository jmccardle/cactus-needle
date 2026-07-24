# Needle → GGUF / ONNX: feasibility investigation

Scope: what it would actually take to run the released 26M Needle checkpoint outside the Cactus
runtime — under llama.cpp (GGUF) and under ONNX Runtime. Feeds ROADMAP.md "Later / speculative".

Status: investigation only. No code written.

---

## 0. Correction to the premise: the architecture *is* KV-cacheable

The claim "no KV cache" is true of `needle/model/run.py`, not of the architecture.

`run.py:145-168` regenerates by re-running the **entire decoder over the full padded 512-slot
buffer at every step** — `decode_fn(params, dec_buffer, encoder_out, enc_mask)` inside the token
loop. That is a reference implementation written for clarity, not a constraint of the model.

The decoder is an ordinary causal stack: pre-norm → masked self-attn (RoPE, absolute positions) →
gated residual → cross-attn (no RoPE) → gated residual. Nothing reads future tokens; nothing is
position-relative. Standard incremental decoding applies.

`docs/simple_attention_networks.md:62` says the quiet part out loud — *"No input tokens in KV
cache. Encoder-decoder uses a fixed-size encoder representation for cross-attention"* — i.e. the
architecture was **designed** around a cache split:

| state | recomputed per step? | size |
|---|---|---|
| encoder hidden states | no — once per request | `enc_len × 512 × 2B` (1 MB @ 1024) |
| cross-attn K/V (8 layers) | no — once per request | `8 KB × enc_len` (8 MB @ 1024) |
| decoder self-attn K/V (8 layers) | no — appended per token | `8 KB × gen_len` (4 MB @ 512) |

Rough FLOP delta for one decode step, `enc_len=1024`: current JAX path ≈ **13 GFLOP**
(full 512-position decoder + cross-K/V reprojected over all 1024 encoder positions, every step);
a properly cached step ≈ **40 MFLOP**. Order of ~300×. That gap is most of the distance between
this repo and Cactus's quoted 1200 tok/s decode.

> **Measured caveat.** ~300× is the *FLOP* ratio, not the wall-clock speedup. Once implemented
> (`experiments/kv_bench`), the KV-cache delivered **6.6× gen tok/s on CPU** and only **1.1× on
> GPU (RTX 4090) at batch 1** — because a full-buffer pass is only ~0.6 ms on-device, so the
> saved FLOPs are already near-free and wall time is host/dispatch bound. The FLOP win is only
> realized as throughput under **batching** (and would compound with a batched cache). See
> `GPU_NOTES.md`.

**This matters for the export question because it is already solved upstream** — see §1.

---

## 1. The single most important finding: a PyTorch reference already exists

Not in this repo. In the runtime repo, `cactus-compute/cactus`:

- `python/cactus/models/needle/modeling_needle.py` (429 lines) — `NeedleForCausalLM`, HF-style
- `python/cactus/models/needle/configuration_needle.py` — `NeedleConfig`, `model_type = "needle"`
- `python/cactus/models/needle/tokenization_needle.py`
- `python/tests/test_encoder_cross_kv_route.py`
- `cactus-engine/tests/test_needle.cpp`

And the HF repo `Cactus-Compute/needle` already ships `model.safetensors` (228 tensors, BF16,
HF naming) plus a `config.json` with `architectures: ["NeedleForCausalLM"]`, alongside the
`.pkl`. Verified against the released weights:

- `model.embed_tokens.weight` and `lm_head.weight` are byte-identical (genuinely tied; the
  `sqrt(d_model)` embed scale is applied at runtime, not folded into either tensor)
- naming is `model.{encoder,decoder}.layers.N.{self_attn,encoder_attn}.{q,k,v,out}_proj.weight`,
  `.{q,k}_norm.weight`, `input_layernorm.weight`, `encoder_attn_layer_norm.weight`,
  `{attn,self_attn,cross_attn}_gate`, `model.encoder.final_norm.weight`, `model.decoder.norm.weight`

Sidenote for ROADMAP §1: **upstream already publishes safetensors.** The fork's pickle-migration
item is partly obsolete for the *released* checkpoint — what remains is the training/finetune
save-path inside this repo.

`modeling_needle.py` also already implements the cached decode as a **three-graph decomposition**,
which is exactly the shape both export targets want:

```
cactus_source_encode(input_ids, attention_mask)      -> encoder_hidden_states, encoder_attention_mask
cactus_decoder_cross_kv(encoder_hidden_states, mask) -> (cross_k_i, cross_v_i) × 8
cactus_decoder_step(decoder_input_ids, position_ids, -> logits
                    encoder_attention_mask, *cross_kv)
```

One caveat, and it's the main gap for ONNX: `cactus_decoder_step` passes `attention_mask=None`
into self-attention and threads **no self-KV through the signature**. The test asserts
`graph_meta["use_internal_kv_cache"] is True` — the Cactus engine splices the decoder self-attn
cache in *below* the traced graph. ONNX has no such runtime contract, so self-K/V must be promoted
to explicit graph inputs/outputs. That is a ~30-line change to `cactus_decoder_step`.

### The PyTorch class is an inference/export artifact, not a training one

Worth being explicit about, because it determines where fine-tuning has to live:

- `NeedleForCausalLM.forward` returns `Seq2SeqLMOutput(logits=...)` — **no `labels` argument and
  no loss**. HF `Trainer` will not work unmodified.
- It subclasses `PreTrainedModel` only, not `GenerationMixin` (transformers 5.x split these), and
  defines no `prepare_inputs_for_generation`. **`.generate()` is not available.**
- No dropout modules anywhere (the JAX config carries `dropout_rate=0.1`).
- `_add_clipped`'s ±65500 clamp is an fp16 inference guard; under autograd it zeroes gradients
  where it saturates.

Also: the HF `config.json` has **no `auto_map`**, and the modeling code is not in the HF repo. So
`AutoModel.from_pretrained("Cactus-Compute/needle", trust_remote_code=True)` will **not** work —
you must `pip install cactus` and import `NeedleForCausalLM` directly.

### Missing link: there is no published `.pkl` → safetensors converter

Checked the full `cactus-compute/cactus` tree. `python/cactus/convert/` handles HF→Cactus and
`python/cactus/transpile/capture_jax.py` captures JAX *graphs*, but nothing converts Flax
checkpoint **weights** to the HF safetensors layout. The published `model.safetensors` was made by
an unreleased script.

Consequence: **a JAX fine-tune cannot reach ONNX without a converter you write.** It is mechanical
— transpose each Flax `Dense` kernel `(in, out) → (out, in)`, split the `nn.scan` leading axis
(12 enc / 8 dec) into per-layer tensors, rename to the HF keys in §1, drop the contrastive head —
roughly 100 lines. But it is load-bearing for every downstream path and nobody has written it.

---

## 2. What is actually non-standard in this model

Established by reading `needle/model/architecture.py` and dumping `/storage/needle-e1/weights/needle.pkl`
(26,315,421 params, stored fp16, `nn.scan`-stacked with a leading layer axis of 12 / 8):

| feature | where | portability |
|---|---|---|
| **No FFN at all** (`no_feedforward=True`) | every block | trivial — omit tensors. `d_ff=2048` in the config is vestigial; there are no `gate/up/down_proj` tensors in the checkpoint. The matryoshka slicing in `needle/model/export.py` is dead code for this checkpoint. |
| **ZCRMSNorm** `(1+γ)·x/rms` | everywhere | fold at conversion: store `weight = 1+γ`. Exactly what the Gemma converters do. ε is inside the sqrt, matching `ggml_rms_norm` and ONNX `SimplifiedLayerNormalization`. |
| **Gated residual** `x + σ(g)·attn(x)` | per attn block, scalar `g` | fold at conversion: `out_proj *= sigmoid(g)`. Exactly equivalent at inference (dropout is identity). **Costs zero graph changes.** 20 scalars disappear. |
| **Per-head Q/K RMSNorm** over head_dim=64 | every attention | same as Qwen3/Gemma3 QK-norm. Applied *before* RoPE, on the un-repeated K. Standard. |
| **RoPE half-split** (`x[:half]`, `x[half:]`) | enc self-attn, dec self-attn | NeoX convention → `GGML_ROPE_TYPE_NEOX`. Not applied in cross-attn. |
| **GQA 8H/4KV** | all | standard; JAX repeats K/V *before* RoPE, PyTorch after — mathematically identical. |
| **`embed_scale = sqrt(512)`** on both enc and dec inputs, **not** on the tied output | `architecture.py:332` | cannot fold into the tied table. llama.cpp: use `LLM_KV_EMBEDDING_SCALE` (exists). ONNX: a literal Mul. |
| **12 encoder / 8 decoder layers** | asymmetric | llama.cpp: `LLM_KV_DECODER_BLOCK_COUNT` (exists; asymmetric enc/dec was specifically fixed for T5 in Sep 2025). |
| **Contrastive head** (512→128→128, ~82K params) | `contrastive_hidden/proj`, `log_temp` | not needed for tool-call inference — drop from both exports. (Relevant to ROADMAP §3: the head *does* exist in the released weights. Note `architecture.py:385-386` warns pretrain decays it toward 0 — check before betting ToolRAG on it.) |
| **Encoder input format** | `run.py:92-103` | `query_tokens + [<tools>=5] + tools_json_tokens`, truncated to 1024. Decoder primed with `[EOS]=1` (`decoder_start_token_id: 1`). Not a chat template — no llama.cpp chat template will produce it. |
| **Trie-based constrained decoding** | `needle/model/constrained.py` | 409 lines of Python, character-level trie over tool names / arg keys. **Not part of the model.** Every quality number in this repo assumes it is on. Any export that drops it will look worse than the published eval. |

Nothing here is exotic. There is no custom op, no state-space layer, no relative position bias.
Every primitive maps to something both GGML and ONNX already have.

---

## 3. ONNX path — recommended, ~1 week

### Plan
1. `pip install cactus`, load `Cactus-Compute/needle` safetensors into `NeedleForCausalLM`. Verify
   logit parity against the JAX path on a fixed batch (this is the gate — do it first).
2. Fork `cactus_decoder_step` into an export-friendly variant that threads self-KV explicitly:
   `(decoder_input_ids, position_ids, encoder_attention_mask, *cross_kv, *past_self_kv) -> (logits, *present_self_kv)`.
3. Export three graphs, the standard seq2seq layout Optimum uses for T5/Whisper:
   - `encoder_model.onnx` — dynamic axes `{batch, enc_len}`
   - `decoder_cross_kv.onnx` — one-shot, 16 outputs
   - `decoder_with_past.onnx` — dynamic axes `{batch, past_len}`, `dec_len` fixed at 1
   Optionally a fourth `decoder_prefill.onnx` with `past_len=0`, or just guard with an `If`.
4. Reimplement the driver loop (greedy argmax + `constrained.py`, which is pure NumPy and ports
   unchanged) in ~150 lines against `onnxruntime`.

### Known gotchas
- **`enable_gqa=True` in `F.scaled_dot_product_attention`** (`modeling_needle.py:122`) — replace
  with an explicit `repeat_interleave` on K/V before export. The GQA flag does not reliably lower
  through the ONNX exporter.
- **BF16.** ORT CPU has no usable bf16 kernels. Cast to fp32 for export; the model is 26M params
  so fp32 is 105 MB and still trivial. Quantize afterward.
- **`_add_clipped`** (`modeling_needle.py:78`, clamps residuals to ±65500) is an fp16-overflow
  guard. It exports fine as `Clip`; keep it — it is load-bearing if you later go fp16.
- **RMSNorm in fp32.** Both implementations upcast; preserve that or you will see drift on the
  26M model, which has no FFN to absorb error.
- **Do not use `jax2tf`.** It works, but `nn.scan` lowers to `tf.while_loop` → ONNX `Loop`, and
  jax2tf is in maintenance mode. The PyTorch route is strictly better and already written.

### Quantization
ORT dynamic INT8 on the MatMuls. The model was trained with INT4 QAT
(`needle/model/quantize.py`: symmetric, group-32 along the **input** dim), so it should tolerate
INT8 with near-zero loss. Expect ~28 MB.

### Verdict
Low risk. The hard part (a correct PyTorch reimplementation with the cache split) is done and
tested upstream. The remaining work is mechanical plus a driver loop.

---

## 4. GGUF / llama.cpp path — feasible but a genuine upstream contribution, 2–4 weeks

llama.cpp *does* have encoder-decoder machinery. Confirmed present on current master:

- `llama_encode()` in `include/llama.h`, documented as *"processes the batch using the encoder.
  Can store the encoder output internally for later use by the decoder's cross-attention layers"*
- `llama_model_has_encoder()`, `llama_model_has_decoder()`, `llama_model_decoder_start_token()`
- `LLM_ARCH_T5`, `LLM_ARCH_T5ENCODER` in `src/llama-arch.h`
- `LLM_TENSOR_DEC_CROSS_ATTN_{NORM,Q,K,V,OUT}`, `LLM_TENSOR_ENC_ATTN_*`, `LLM_TENSOR_DEC_ATTN_*`
- `LLM_KV_DECODER_BLOCK_COUNT`, `LLM_KV_EMBEDDING_SCALE`, `LLM_KV_ATTENTION_SCALE`
- in `src/llama-graph.h`: `struct llama_cross` (holds `v_embd`, `n_enc`, `seq_ids_enc`),
  `llm_graph_input_cross_embd`, and a dedicated `build_attn(llm_graph_input_attn_cross *, ...)`

So the primitives exist. The work:

### Required changes
1. **`gguf-py/gguf/constants.py`** — `MODEL_ARCH.NEEDLE`, tensor name table.
2. **`src/llama-arch.{h,cpp}`** — `LLM_ARCH_NEEDLE` + `LLM_TENSOR_NAMES` entry. **Six new tensor
   enums** are needed: `{ENC,DEC}_ATTN_{Q,K}_NORM` and `DEC_CROSS_ATTN_{Q,K}_NORM`. T5 has no
   QK-norm, so the enc/dec tensor families don't cover it. (You *can* fold the learned `(1+γ)`
   into the `q_proj`/`k_proj` rows — it's a diagonal post-projection scale — but the `÷RMS` itself
   is nonlinear and must stay in the graph, so you need the op regardless. Not worth the trick.)
3. **`src/llama-model.h`** — add the QK-norm fields to `llama_layer`'s enc/dec section.
4. **`src/llama-model.cpp`** — `load_hparams` (rope type NEOX, `f_embedding_scale = sqrt(512)`,
   `f_attention_scale = 1/8`), `load_tensors`, and a `llm_build_needle` graph struct. The graph is
   *simpler* than T5's: no FFN, no relative position bias, RoPE instead. Register in the
   `build_graph` and `llama_model_rope_type` switches.
5. **`convert_hf_to_gguf.py`** — `@ModelBase.register("NeedleForCausalLM")`. In `modify_tensors`:
   add 1.0 to every norm weight, multiply each `out_proj` by `sigmoid(gate)` and drop the gate,
   drop the contrastive head, drop the duplicate `lm_head` (tied). Vocab via
   `_set_vocab_sentencepiece()` — `tokenizer.model` is on the HF repo and the 6 specials
   (`PAD=0, EOS=1, BOS=2, UNK=3, <tool_call>=4, <tools>=5`) are fixed IDs.

### Real risks
- **The enc-dec path in llama.cpp is a minority, under-exercised code path.** T5Encoder support was
  broken by a refactor and commented out ([#12588](https://github.com/ggml-org/llama.cpp/issues/12588));
  `llama-server` support for enc-dec is poor. Budget time for fixing things that are already broken
  before your model can even be a fair test. This is the dominant schedule risk, not the model.
- **No usable frontend.** `llama-cli` handles enc-dec awkwardly and `llama-server` largely doesn't.
  Needle's input isn't a chat template — it's `query + <tools> + tools_json` on the encoder. You'd
  likely be writing against `libllama` directly, which erases much of the "just use llama.cpp"
  benefit.
- **Constrained decoding.** `constrained.py` is a character-level trie, not a grammar. Porting it
  to GBNF is doable (llama.cpp has `json-schema-to-grammar`) but it's a separate project, and
  without it you will not reproduce the published accuracy.
- **Upstreaming.** A single-model arch for a 26M non-Llama-family model is a hard sell on
  ggml-org/llama.cpp. Plan on maintaining a fork, or land it only if the SAN family grows.

### Payoff if you do it
Q4_0 maps almost perfectly onto the training-time QAT (symmetric, block-32, along ne[0] = the
input dim — same axis the QAT grouped on). Expect ~15 MB at Q4_0, ~28 MB at Q8_0, 53 MB at F16.

---

## 5. Recommendation

**Do ONNX. Skip GGUF unless something specific requires llama.cpp.**

The reasoning is mostly about what llama.cpp buys you. Its value is the ecosystem — server,
tooling, quant zoo, ubiquity — and for an encoder-decoder model with a non-chat input format,
you get almost none of that. You'd be writing a custom `libllama` driver anyway, and paying a
2–4 week arch-implementation tax plus ongoing fork maintenance to get there. Meanwhile the GGML
kernel advantage is thin: at 26M params with no FFN, this is a memory-bandwidth-trivial model.

ONNX gets you deployment outside the Cactus stack, on CPU and on the GPU server, in about a week,
on top of an upstream PyTorch implementation that is already written, already tested, and already
implements the cache split the architecture was designed for.

Suggested sequencing:
1. **Write the `.pkl` → safetensors converter** (§1) — it does not exist anywhere, everything
   downstream needs it, and it is the same flatten/rename work ROADMAP §1 already wants.
2. **Parity harness** — JAX vs PyTorch logits on a fixed batch, through that converter. Cheap,
   de-risks everything after it, and produces the fixed-eval-batch gate ROADMAP §1 asks for.
3. **KV-cached decode + port `constrained.py`** — the ~300× decode win. Do this in whichever
   framework you plan to serve from; it is worth doing even if you never export.
4. **ONNX export (3–4 graphs) + ORT driver loop + INT8.**
5. Revisit GGUF only if a deployment target genuinely mandates llama.cpp.

Fine-tuning stays in JAX on the `.pkl` throughout — that is where the data pipeline, sequence
packing, eval, and the playground live, and the PyTorch class has no loss, no dropout, and no
`generate()`. Treat PyTorch/safetensors as the export target and the converter as the bridge.

Also worth folding back into ROADMAP: the released checkpoint's `config.json` still has no
`max_position_embeddings`, but the `.pkl` config carries `max_seq_len: 1024`, matching
`DEFAULT_MAX_ENC_LEN = 1024`. That's the answer ROADMAP §2 is looking for — worth confirming
empirically, since RoPE θ=10000 at d_head=64 won't degrade gracefully past its training length.

---

## Sources

- [llama.cpp: T5 (encoder-decoder) support](https://github.com/ggml-org/llama.cpp/issues/5763)
- [llama.cpp #12588: T5Encoder support broken](https://github.com/ggml-org/llama.cpp/issues/12588)
- [llama.cpp #8900: T5-based encoder-only models](https://github.com/ggml-org/llama.cpp/issues/8900)
- [cactus-compute/cactus](https://github.com/cactus-compute/cactus)
- [Cactus-Compute/needle on HF](https://huggingface.co/Cactus-Compute/needle)
- [Running encoder-decoder models with llama.cpp](https://huggingface.co/Felladrin/gguf-sharded-LaMini-Flan-T5-248M/discussions/1)
