"""Parity gate: KV-cached greedy decode must match stock full-buffer greedy decode."""
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llguidance_poc"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, jax, jax.numpy as jnp

from kv_model import KVNeedle
from needle_llg import NeedleLLG
from tools_spec import tools_json_all

DATA = os.path.join(os.path.dirname(__file__), "..", "llguidance_poc", "data")


def kv_greedy(kv, enc_out, enc_mask):
    ck, cv, cbias = kv.cross_kv(enc_out, enc_mask)
    kc, vc = kv.fresh_cache()
    eos = kv.tok.eos_token_id
    tok = eos
    gen = []
    logits, kc, vc = kv.extend([tok], 0, kc, vc, ck, cv, cbias)
    for pos in range(1, kv.max_dec):
        nt = int(jnp.argmax(logits[-1]))
        if nt == eos:
            break
        gen.append(nt)
        logits, kc, vc = kv.extend([nt], pos, kc, vc, ck, cv, cbias)
    text = kv.tok.decode(gen)
    if text.startswith("<tool_call>"):
        text = text[len("<tool_call>"):]
    return text.strip(), gen


def main():
    kv = KVNeedle(max_dec=64)
    stock = NeedleLLG(max_gen=64)
    tools = tools_json_all()
    domains = json.load  # noqa
    n_match = n = 0
    diffs = []
    import glob
    for f in sorted(glob.glob(os.path.join(DATA, "*.json"))):
        for ex in json.load(open(f))[:5]:  # 5 per domain = 50 total
            q = ex["query"]
            eo, em, _ = kv.encode(q, tools)
            kv_text, _ = kv_greedy(kv, eo, em)
            eo2, em2 = stock.encode(q, tools)
            st = stock.gen_unconstrained(eo2, em2)
            n += 1
            if kv_text == st["text"]:
                n_match += 1
            else:
                diffs.append((q[:40], st["text"][:60], kv_text[:60]))
    print(f"\nPARITY: {n_match}/{n} exact greedy match")
    for q, a, b in diffs[:12]:
        print(f"  DIFF q={q!r}\n     stock={a!r}\n     kv   ={b!r}")


if __name__ == "__main__":
    main()
