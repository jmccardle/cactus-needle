"""4-config decode benchmark: stock / KV / JumpForward / KV+JumpForward.

Reports prefill throughput (encoder tokens/s) and generation throughput (output tokens/s),
plus mean model forward passes per call. Stock & KV produce the same (unconstrained) output;
JF & KV+JF produce the same (grammar-constrained) output — so each pair isolates one axis:
  stock->KV      : KV-cache win (per-token cost)
  stock->JF      : jump-forward win (fewer passes)
  JF->KV+JF      : KV-cache win under jump-forward
  KV->KV+JF      : jump-forward win under KV-cache
"""
import os, sys, json, time, glob, statistics as st
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llguidance_poc"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, jax, jax.numpy as jnp

from kv_model import KVNeedle
from needle_llg import NeedleLLG
from tools_spec import tools_json_all
from llguidance import LLMatcher
from llguidance.numpy import allocate_token_bitmask, fill_next_token_bitmask

DATA = os.path.join(os.path.dirname(__file__), "..", "llguidance_poc", "data")
OUT = os.path.dirname(__file__)
N_PER_DOMAIN = 4          # prompts per domain used for timing
MAX_DEC = 64


def now():
    return time.perf_counter()


# ---------- KV decode drivers ----------

def kv_free(kv, ck, cv, cbias):
    """KV-cached unconstrained greedy. Returns (text, gen_tokens, passes, gen_seconds)."""
    kc, vc = kv.fresh_cache()
    eos = kv.tok.eos_token_id
    t = now()
    logits, kc, vc = kv.extend([eos], 0, kc, vc, ck, cv, cbias)
    passes = 1
    gen = []
    for pos in range(1, kv.max_dec):
        nt = int(jnp.argmax(logits[-1]))
        if nt == eos:
            break
        gen.append(nt)
        logits, kc, vc = kv.extend([nt], pos, kc, vc, ck, cv, cbias)
        passes += 1
    dt = now() - t
    return _finish(kv, gen), gen, passes, dt


def kv_jf(kv, nl, ck, cv, cbias):
    """KV-cached + jump-forward. nl = NeedleLLG (for grammar/matcher/prefix)."""
    kc, vc = kv.fresh_cache()
    eos = kv.tok.eos_token_id
    mask_buf = allocate_token_bitmask(1, nl.lltok.vocab_size)
    t = now()
    logits, kc, vc = kv.extend([eos], 0, kc, vc, ck, cv, cbias)
    passes = 1
    t0 = int(jnp.argmax(logits[-1]))       # <tool_call> (free)
    gen = [t0]
    pos = 1
    logits, kc, vc = kv.extend([t0], pos, kc, vc, ck, cv, cbias); passes += 1
    m = LLMatcher(nl.lltok, nl.grammar)
    # inject invariant ' [{"name":"' prefix as one batched extend (jump-forward)
    pref = list(nl.prefix_tokens)
    m.consume_tokens(pref)
    pos += 1
    logits, kc, vc = kv.extend(pref, pos, kc, vc, ck, cv, cbias); passes += 1
    gen += pref; pos += len(pref) - 1
    while pos < kv.max_dec - 1:
        ff = m.compute_ff_tokens()
        if ff:
            run, stop = [], False
            for x in ff:
                if x == eos:
                    stop = True; break
                run.append(x)
            if run:
                m.consume_tokens(run)
                logits, kc, vc = kv.extend(run, pos + 1, kc, vc, ck, cv, cbias); passes += 1
                gen += run; pos += len(run)
            if stop or m.is_stopped():
                break
            continue
        if m.is_stopped():
            break
        fill_next_token_bitmask(m, mask_buf, 0)
        allowed = np.unpackbits(mask_buf.view(np.uint8), bitorder="little")[:kv.emb.shape[0]]
        row = np.array(logits[-1], dtype=np.float32)
        row[allowed == 0] = -np.inf
        if not np.isfinite(row).any():
            break
        nt = int(np.argmax(row))
        m.consume_token(nt)
        logits, kc, vc = kv.extend([nt], pos + 1, kc, vc, ck, cv, cbias); passes += 1
        gen.append(nt); pos += 1
        if m.is_stopped():
            break
    jnp.asarray(logits).block_until_ready()  # forced-run extends dispatch without a read; await
    dt = now() - t
    return _finish(kv, gen), gen, passes, dt


def _finish(kv, gen):
    text = kv.tok.decode(gen)
    if text.startswith("<tool_call>"):
        text = text[len("<tool_call>"):]
    return text.strip()


def main():
    kv = KVNeedle(max_dec=MAX_DEC)
    nl = NeedleLLG(max_gen=MAX_DEC)
    tools = tools_json_all()

    prompts = []
    for f in sorted(glob.glob(os.path.join(DATA, "*.json"))):
        for ex in json.load(open(f))[:N_PER_DOMAIN]:
            prompts.append(ex["query"])

    # ---- warmup: compile every jitted shape (full-buffer paths + extend for m=1..24) ----
    eo, em, _ = kv.encode(prompts[0], tools)
    ck, cv, cbias = kv.cross_kv(eo, em)
    nl.gen_unconstrained(eo, em); nl.gen_constrained(eo, em)
    kc, vc = kv.fresh_cache()
    for mm in range(1, 25):
        kv.extend(list(range(mm)), 0, kc, vc, ck, cv, cbias)
    kv_free(kv, ck, cv, cbias); kv_jf(kv, nl, ck, cv, cbias)

    acc = {c: {"prefill_toks": 0, "prefill_s": 0.0, "gen_toks": 0, "gen_s": 0.0,
               "passes": [], "outlen": []} for c in ["stock", "kv", "jf", "kv_jf"]}

    for q in prompts:
        # prefill: encoder (shared) + cross_kv (KV configs)
        t = now(); eo, em, enc_len = kv.encode(q, tools); t_enc = now() - t
        t = now(); ck, cv, cbias = kv.cross_kv(eo, em); jnp.asarray(ck).block_until_ready(); t_cross = now() - t

        # stock (full-buffer, unconstrained)
        r = nl.gen_unconstrained(eo, em)
        _rec(acc["stock"], enc_len, t_enc, r["tokens"], r["ms"] / 1000, r["passes"])
        # JF (full-buffer, grammar + jump-forward)
        r = nl.gen_constrained(eo, em)
        _rec(acc["jf"], enc_len, t_enc, r["tokens"], r["ms"] / 1000, r["passes"])
        # KV (cached, unconstrained)
        _txt, gen, passes, dt = kv_free(kv, ck, cv, cbias)
        _rec(acc["kv"], enc_len, t_enc + t_cross, len(gen), dt, passes)
        # KV + JF (cached, grammar + jump-forward)
        _txt, gen, passes, dt = kv_jf(kv, nl, ck, cv, cbias)
        _rec(acc["kv_jf"], enc_len, t_enc + t_cross, len(gen), dt, passes)

    write_report(acc, len(prompts))


def _rec(a, enc_len, prefill_s, gen_toks, gen_s, passes):
    a["prefill_toks"] += enc_len
    a["prefill_s"] += prefill_s
    a["gen_toks"] += gen_toks
    a["gen_s"] += gen_s
    a["passes"].append(passes)
    a["outlen"].append(gen_toks)


def write_report(acc, n):
    names = {"stock": "stock (full-buffer)", "kv": "KV-cache",
             "jf": "JumpForward", "kv_jf": "KV + JumpForward"}
    backend = jax.default_backend().upper()
    dev = jax.devices()[0]
    devname = getattr(dev, "device_kind", str(dev))
    lines = ["# Needle decode benchmark: stock / KV / JumpForward / KV+JumpForward\n"]
    lines.append(f"{n} prompts (all 10 tools shown each call), greedy, {backend} JAX "
                 f"({devname}). Prefill = encoder ingest (+cross-KV projection for KV configs). "
                 f"Generation = output-token decode loop.\n")
    lines.append("| config | prefill tok/s | **gen tok/s** | mean fwd passes | mean out tok | "
                 "mean total ms | end-to-end speedup |")
    lines.append("|---|---|---|---|---|---|---|")
    base_gen = None
    base_total = None
    for c in ["stock", "kv", "jf", "kv_jf"]:
        a = acc[c]
        pf = a["prefill_toks"] / a["prefill_s"]
        gen = a["gen_toks"] / a["gen_s"]
        total_ms = 1000 * (a["prefill_s"] + a["gen_s"]) / n
        if c == "stock":
            base_gen = gen; base_total = total_ms
        lines.append(f"| {names[c]} | {pf:,.0f} | **{gen:,.0f}** | "
                     f"{st.mean(a['passes']):.1f} | {st.mean(a['outlen']):.1f} | "
                     f"{total_ms:.0f} | {base_total / total_ms:.1f}x |")
    # pairwise decode-time deltas
    def gens(c):
        return acc[c]["gen_toks"] / acc[c]["gen_s"]
    lines.append("\n## Reading the axes\n")
    lines.append(f"- **KV-cache win** (same unconstrained output): stock -> KV = "
                 f"{gens('kv') / gens('stock'):.1f}x gen tok/s.")
    lines.append(f"- **Jump-forward win** (same full-buffer path): stock -> JF = "
                 f"{gens('jf') / gens('stock'):.1f}x gen tok/s (fewer model passes on the same output).")
    lines.append(f"- **Combined**: stock -> KV+JF = {gens('kv_jf') / gens('stock'):.1f}x gen tok/s.")
    lines.append(f"- KV-cache under jump-forward: JF -> KV+JF = {gens('kv_jf') / gens('jf'):.1f}x.")
    lines.append(f"- Jump-forward under KV-cache: KV -> KV+JF = {gens('kv_jf') / gens('kv'):.1f}x.")
    r_kv = gens('kv') / gens('stock')
    r_jf = gens('jf') / gens('stock')
    r_comb = gens('kv_jf') / gens('stock')
    pred = r_kv * r_jf
    best = max(r_kv, r_jf)
    if pred and r_comb >= 0.8 * pred:
        lines.append(f"- The two wins are ~orthogonal and multiply: {r_kv:.1f}x x {r_jf:.1f}x "
                     f"~= {r_comb:.1f}x. Each attacks a different cost (KV = per-pass work, "
                     f"jump-forward = number of passes), so on this backend they compound.")
    elif r_comb >= best * 1.03:
        lines.append(f"- The two wins **help but do not multiply** ({r_kv:.1f}x x {r_jf:.1f}x would "
                     f"predict {pred:.1f}x; actual stock -> KV+JF is {r_comb:.1f}x). They overlap: "
                     f"KV-cache collapses per-pass cost, so once a pass is cheap, cutting the "
                     f"*number* of passes adds less than its standalone {r_jf:.1f}x.")
    else:
        lines.append(f"- **Combining is not additive here — the best single config wins.** "
                     f"stock -> KV = {r_kv:.1f}x, stock -> JF = {r_jf:.1f}x, but stock -> KV+JF is "
                     f"only {r_comb:.1f}x, *below* JF alone (JF -> KV+JF = {gens('kv_jf')/gens('jf'):.1f}x). "
                     f"On this backend a full-buffer pass is already compute-negligible, so the "
                     f"KV-cache optimizes work that is already free while adding per-pass kernel/host "
                     f"overhead (dynamic-slice cache writes, more small ops, grammar consume). "
                     f"The bottleneck has shifted from per-pass FLOPs (compute-bound, CPU) to "
                     f"per-pass dispatch/host latency (launch-bound, GPU batch-1) — where KV-cache "
                     f"and jump-forward stop compounding and batching (see batched_throughput) is "
                     f"the lever instead.")
    stock_ms = 1000 * (acc["stock"]["prefill_s"] + acc["stock"]["gen_s"]) / n
    kv_ms = 1000 * (acc["kv"]["prefill_s"] + acc["kv"]["gen_s"]) / n
    if kv_ms <= stock_ms * 0.9:
        lines.append("\n**Prefill vs generation tradeoff.** KV's prefill tok/s is *lower* (it projects "
                     "cross-K/V once, up front) but that is exactly the per-token work stock repeats "
                     f"every step — so end-to-end latency drops ({stock_ms:.0f} -> {kv_ms:.0f} ms; see "
                     "'mean total ms'). The cross-KV front-load pays for itself within the first "
                     "couple of output tokens.")
    else:
        lines.append("\n**Prefill vs generation tradeoff.** KV's prefill tok/s is *lower* (it projects "
                     "cross-K/V once, up front). On this backend that front-load does **not** pay off: "
                     f"end-to-end latency is flat across configs ({stock_ms:.0f} vs {kv_ms:.0f} ms) "
                     "because the un-cached full-buffer pass it eliminates is already ~free here — the "
                     "wall time is host-loop/dispatch bound, not model-compute bound.")
    lines.append("\nNotes: mean-fwd-passes for stock/KV = output tokens + 1 (one pass per token); "
                 "for JF/KV+JF it is the number of *model* passes after collapsing grammar-forced "
                 f"runs into batched extends. Absolute tok/s are {backend}-bound and un-cached "
                 "per-pass; the *ratios* are the result. On the un-cached decode each stock pass re-runs the "
                 "whole decoder over the full buffer AND re-projects cross-K/V over all encoder "
                 "positions — that reprojection is exactly what the KV config caches once at prefill.")
    suffix = "" if jax.default_backend() == "cpu" else "_" + jax.default_backend()
    open(os.path.join(OUT, f"results{suffix}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
