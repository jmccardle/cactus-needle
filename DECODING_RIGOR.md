# Needle constrained decoding: rigor + efficiency for a robotics target

Question behind this: can Needle's tool-call output be *trusted* enough to drive an actuator, and
can it be made both more rigorous (only valid values) and more efficient (skip forced tokens)?

Short version: the current constrained decoder is **not** rigorous enough for robotics, for a
specific and fixable reason. The fix (schema→grammar) and the efficiency win (jump-forward) are
two different mechanisms that happen to share one engine: **llguidance**.

---

## 1. What Needle constrains today — and the gap

`needle/model/constrained.py` (a char-level trie + JSON state machine, driven from
`run.py`'s greedy-argmax loop) constrains exactly two spans:

- **tool names** — after `"name":"`, masked to known tool names
- **argument keys** — after `"arguments":{"`, masked to that tool's param names

Its own module docstring is explicit: *"Argument values are unconstrained (strings, numbers,
booleans, objects)."*

For robotics this is backwards. Names and keys are the **least** safety-critical part — a wrong
key usually just fails a lookup. The **values** are what move the robot, and they are exactly what
is left free. Concretely, today's decoder does **not** enforce:

- **types** — a `{"type":"number"}` param can decode as `"left"`; a boolean can decode as `7`
- **enums** — `{"enum":["forward","backward","left","right"]}` is not honored at all; the model
  can emit any string
- **ranges / structure** — no bound on a coordinate, no guarantee required args are present, no
  block on extra keys
- **hard failure** — `constrain_logits`/`apply_constraints` **fall back to unconstrained** (return
  raw logits + a warning) whenever the trie goes off-path or masks everything. So even the
  name/key guarantees are *soft*: any edge case silently degrades to free generation. A robot
  wants abstain-or-halt here, not "guess freely."

Plus it's greedy argmax only (deterministic — fine) and Python-per-token (slow — see §3).

**Verdict:** trustworthy for *routing* (pick the right tool), not for *arguments*. The part a
robot depends on is the part that's unconstrained and soft-failing.

---

## 2. Rigor: compile the tool schema into a grammar (llguidance / GBNF)

The fix is to stop hand-rolling a trie over two spans and instead compile each call's JSON Schema
into a grammar that constrains the **whole** object — structure, keys, *and* typed values. Needle's
tool format (`parameters.{key}.{type, description, required, enum?}`) is already JSON-Schema-shaped,
so this is a direct compile.

What a schema grammar buys, per value kind:

| param kind | grammar guarantee | robotics use |
|---|---|---|
| **enum** `["forward","backward","left","right"]` | output is exactly one of the literals — **zero free generation** | this is the answer for every discrete command. Fully rigorous regardless of model behavior. |
| **boolean / null** | `true`/`false`/`null` only | flags, gripper open/close |
| **number / integer** | numeric syntax (sign, digits, one `.`) | coordinates, joint IDs. **Caveat:** syntax ≠ range. `[0, 3.14]` is not cleanly grammar-expressible; validate bounds downstream or discretize. |
| **string (free)** | quoted, escaped, terminated | non-actuating fields only (a message to speak, a label). Never let a free string reach an actuator. |
| **object / array** | required keys present, no extras, correct nesting | whole-call well-formedness |

The design principle for the robotics case: **the grammar is your safety boundary, not the model.**
A schema-constrained decode makes malformed or out-of-vocabulary commands *structurally
impossible* — a hard guarantee that holds under distribution shift, unlike "fine-tune until it
seems reliable," which is only ever statistical. Keep every safety-critical argument an **enum or a
bounded/discretized value** and the 26M model can only ever choose *among legal actions*, never
invent one.

**Fine-tuning is still needed — it's complementary, not redundant.** The grammar guarantees the
output is *valid*; it does not guarantee it's *correct*. The model still has to put probability
mass on the *right* enum branch. So: grammar = validity/safety floor (hard), fine-tune = accuracy
(soft). Do both. And note the two interact well — under a grammar you can fine-tune purely on
picking the right branch, since the model no longer has to spend capacity learning JSON syntax.

**Engine choice: llguidance.** It's the grammar engine your own llama.cpp commit already drives
(`llg_matcher_*`). It's a Rust core with a C API **and Python bindings** (the `llguidance` package,
from the `guidance` project), so it is reusable *outside* llama.cpp — including a JAX or ONNX driver
loop for Needle. It computes per-step token masks (the rigor part) **and** forced-token runs (the
efficiency part, §3) from the same matcher. llama.cpp's `json-schema-to-grammar` (GBNF) is an
alternative for the schema→grammar step if you stay in-runtime.

**Abstention tie-in (ROADMAP §4):** a grammar cannot force the model to *not* call — if the only
legal productions are calls, the robot always gets one. So the grammar must itself admit the
no-op: include a `no_applicable_operation` tool (or an empty-array production) as a legal branch.
Grammar rigor and abstention are the same work here.

---

## 3. Efficiency: jump-forward — and this is literally your commit

Your goal "skip decoding passes that have exactly one legal token" **is** jump-forward decoding,
and `f7c5864c` already implements it:

> When an llguidance grammar forces a run of tokens, inject them into the slot's batch in a single
> decode with only the last forced token producing logits, skipping the forward passes for the
> grammar-certain tokens. […] Byte-identical to the unconstrained path at temperature 0; measured
> 8.6× faster on a mostly-forced grammar on qwen36-35B (4090), 0.1% overhead on free generation.

Two properties from your own commit message that matter for the robotics case:

- **Provably lossless.** It only skips passes where the grammar leaves *exactly one* legal token —
  i.e. no decision to make. "Byte-identical at temp 0" means jump-forward changes *nothing* about
  what's emitted; it only removes forward passes that couldn't have gone any other way. So it is
  pure latency, zero accuracy/trust cost. Your framing is exactly correct.
- **The forced tokens come from the grammar, not the model.** The parts it skips are the parts the
  model was never choosing — which is also *why* they're trustworthy. Efficiency and rigor are the
  same coin: the more the grammar forces, the less the model decides, the faster *and* safer it is.

**Why this is unusually strong for Needle specifically.** Needle's output is short and highly
scaffolded: `[{"name":"`, `","arguments":{"`, `":`, `"}}]`, plus keys and enum values that become
forced the instant they're disambiguated. A large fraction of a ~40-token tool call is
grammar-certain. Under a schema grammar with enums, a command like
`[{"name":"move","arguments":{"direction":"forward"}}]` is almost entirely forced — plausibly
**~5–8 real decode passes instead of ~40**. That compounds with the KV-cache win from
`EXPORT_PATHS.md`: caching cuts the cost *per* pass (~300× in FLOPs), jump-forward cuts the
*number* of passes. (Measured, the two compound to **10.5× gen tok/s on CPU**; on GPU at batch 1
they don't — the per-pass compute is already free and the bottleneck is host dispatch, so the
lever there is batching. See `GPU_NOTES.md`.)

### Where the commit applies vs. where you re-implement the technique

- **If Needle runs under llama.cpp (the GGUF path):** your commit + a schema grammar gives you
  **both rigor and efficiency in one runtime, mostly for free.** This meaningfully re-weights the
  GGUF-vs-ONNX call from `EXPORT_PATHS.md` *for the robotics use case* — the enc-dec porting tax
  buys a working, jump-forward-capable, grammar-constrained server. Still gated on getting Needle's
  enc-dec arch landed in llama.cpp first (non-trivial, see that doc).
- **If Needle runs under JAX/ONNX (recommended elsewhere):** the *commit's C++* doesn't port, but
  the *technique* does, and the *engine* does. Build the driver loop around llguidance's Python
  matcher:
  1. each step, ask the matcher for the token mask; apply it to logits; argmax
  2. after accepting, ask `compute_ff_tokens` for the forced run
  3. append the whole run to the decoder sequence and do **one** batched (prefill-style) decode
     that ingests them, taking logits only at the last position
  This is the same shape as your server commit's `handle_last_sampled_token()` injection, minus the
  slot/batch plumbing. It replaces `constrained.py` wholesale and fixes the soft-fallback problem
  (llguidance hard-stops instead of silently freeing).

---

## 4. Recommendation

1. **Replace `constrained.py` with an llguidance-backed decoder.** This is the load-bearing change:
   it moves the safety boundary from a two-span soft trie to a full schema grammar with hard
   failure, and it's the same engine in every runtime.
2. **Make every safety-critical argument an enum or a bounded/discretized value** in the tool
   schemas. Free strings only for non-actuating fields. Include a `no_applicable_operation` branch.
3. **Fine-tune under the grammar**, on the right-branch decision — grammar handles validity, tuning
   handles accuracy. Evaluate with the grammar *on*, since every published Needle number already
   assumes constrained decoding.
4. **Turn on jump-forward** — free once you're on llguidance. Reuse `f7c5864c` directly if you land
   the llama.cpp path; re-implement the ~10-line inject loop against the llguidance Python matcher
   if you're in JAX/ONNX.
5. **Trust argument:** for robotics, lead with the grammar guarantee (illegal commands are
   structurally impossible), not model reliability. Fine-tuning raises the odds of the *right* legal
   action; the grammar guarantees you never get an *illegal* one — and jump-forward makes the forced
   (already-trustworthy) spans nearly free.

---

## Source

- `turboquant_experiments/repos/upstream-llama-cpp` @ `f7c5864c` —
  "server: jump-forward decoding for llguidance grammars" (John, 2026-07-16)
- `needle/model/constrained.py` — current trie/state-machine constrained decoder
- `EXPORT_PATHS.md` — KV-cache and runtime analysis this builds on
