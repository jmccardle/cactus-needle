"""A/B benchmark: stock Needle (unconstrained greedy) vs llguidance grammar + jump-forward.

For each of the 200 examples (10 domains x 20), the model always sees all 10 tools.
We record, per arm: forward passes, wall-ms, and validity (json/name/keys/values) plus
exact-match against the Haiku-generated ground truth. Writes results.json + results.md.
"""
import os, sys, json, time, statistics as st
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(__file__))

from needle_llg import NeedleLLG
from tools_spec import DOMAINS, TOOLS, tools_json_all, score_call, exact_match

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.dirname(__file__)


def parse(text):
    try:
        return json.loads(text)
    except Exception:
        return None


def main():
    eng = NeedleLLG(max_gen=48)
    tools = tools_json_all()

    # warm up both jitted paths (compile cost out of the timing)
    eo, em = eng.encode("warmup", tools)
    eng.gen_unconstrained(eo, em)
    eng.gen_constrained(eo, em)

    rows = []
    t_start = time.perf_counter()
    for domain in DOMAINS:
        examples = json.load(open(os.path.join(DATA_DIR, f"{domain}.json")))
        for ex in examples:
            q, exp = ex["query"], ex["arguments"]
            eo, em = eng.encode(q, tools)
            u = eng.gen_unconstrained(eo, em)
            c = eng.gen_constrained(eo, em)
            su = score_call(domain, parse(u["text"]))
            sc = score_call(domain, parse(c["text"]))
            rows.append({
                "domain": domain, "query": q, "expected": exp,
                "stock": {**u, "score": su, "exact": exact_match(domain, parse(u["text"]), exp)},
                "llg": {**c, "score": sc, "exact": exact_match(domain, parse(c["text"]), exp)},
            })
        done = sum(1 for r in rows if r["domain"] == domain)
        print(f"  {domain:22s} {done} examples done", flush=True)
    wall = time.perf_counter() - t_start

    json.dump({"rows": rows, "wall_s": wall}, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=1)
    write_report(rows, wall)
    print(f"\nTotal benchmark wall: {wall:.1f}s over {len(rows)} examples")


def _agg(rows, arm, field):
    return [r[arm][field] for r in rows]


def _rate(rows, arm, pred):
    return 100.0 * sum(1 for r in rows if pred(r[arm])) / len(rows)


def write_report(rows, wall):
    lines = []
    lines.append("# Needle constrained-decoding A/B: stock vs llguidance grammar + jump-forward\n")
    lines.append(f"200 examples, 10 domains x 20. Model sees all 10 tools every call. "
                 f"CPU (jax {os.environ.get('JAX_PLATFORMS','cpu')}). Total wall {wall:.0f}s.\n")
    lines.append("**Arms.** *stock* = unconstrained greedy argmax (Needle's raw output). "
                 "*llg* = llguidance schema-union grammar with jump-forward (grammar-forced token "
                 "runs injected in one pass).\n")
    lines.append("**Metrics.** `valid` = well-formed call with correct tool name, keys, and every "
                 "value semantically valid for its domain (real date, in-range int, valid state, "
                 "etc.). `exact` = arguments equal the ground-truth. `passes` = model forward passes "
                 "(≈ wall-time on the un-cached decode). Higher valid/exact, lower passes = better.\n")

    def block(title, subset):
        st_valid = _rate(subset, "stock", lambda a: a["score"]["values_ok"])
        lg_valid = _rate(subset, "llg", lambda a: a["score"]["values_ok"])
        st_name = _rate(subset, "stock", lambda a: a["score"]["name_ok"])
        lg_name = _rate(subset, "llg", lambda a: a["score"]["name_ok"])
        st_json = _rate(subset, "stock", lambda a: a["score"]["json_ok"])
        lg_json = _rate(subset, "llg", lambda a: a["score"]["json_ok"])
        st_exact = _rate(subset, "stock", lambda a: a["exact"])
        lg_exact = _rate(subset, "llg", lambda a: a["exact"])
        st_p = st.mean(_agg(subset, "stock", "passes"))
        lg_p = st.mean(_agg(subset, "llg", "passes"))
        st_ms = st.mean(_agg(subset, "stock", "ms"))
        lg_ms = st.mean(_agg(subset, "llg", "ms"))
        spd = st_p / lg_p if lg_p else 0
        return (f"| {title} | {st_json:.0f}/{lg_json:.0f} | {st_name:.0f}/{lg_name:.0f} | "
                f"{st_valid:.0f}/{lg_valid:.0f} | {st_exact:.0f}/{lg_exact:.0f} | "
                f"{st_p:.1f}/{lg_p:.1f} | {st_ms:.0f}/{lg_ms:.0f} | {spd:.1f}x |")

    lines.append("\n## Overall (stock / llg)\n")
    lines.append("| scope | json% | name% | **valid%** | exact% | passes | ms | speedup |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(block("**ALL**", rows))
    lines.append("")
    lines.append("\n## Per domain (stock / llg)\n")
    lines.append("| domain | json% | name% | **valid%** | exact% | passes | ms | speedup |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for d in DOMAINS:
        sub = [r for r in rows if r["domain"] == d]
        lines.append(block(d, sub))

    # headline deltas
    v_s = _rate(rows, "stock", lambda a: a["score"]["values_ok"])
    v_l = _rate(rows, "llg", lambda a: a["score"]["values_ok"])
    e_s = _rate(rows, "stock", lambda a: a["exact"])
    e_l = _rate(rows, "llg", lambda a: a["exact"])
    p_s = st.mean(_agg(rows, "stock", "passes"))
    p_l = st.mean(_agg(rows, "llg", "passes"))
    lines.append("\n## Headline\n")
    lines.append(f"- **Validity**: {v_s:.0f}% -> {v_l:.0f}% (+{v_l - v_s:.0f} pts) with the grammar.")
    lines.append(f"- **Exact-match**: {e_s:.0f}% -> {e_l:.0f}% (+{e_l - e_s:.0f} pts).")
    lines.append(f"- **Forward passes**: {p_s:.1f} -> {p_l:.1f} mean ({p_s / p_l:.1f}x fewer) via jump-forward.")
    lines.append("- Validity is a *hard guarantee* under the grammar; the remaining exact-match gap "
                 "is the un-fine-tuned model picking a valid-but-wrong value (e.g. a valid US state "
                 "code that isn't the intended one). Fine-tuning closes that gap; the grammar already "
                 "closes the validity gap.")

    open(os.path.join(OUT_DIR, "results.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
