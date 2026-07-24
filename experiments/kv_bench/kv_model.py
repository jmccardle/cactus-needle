"""KV-cached Needle decoder in JAX — the EXPORT_PATHS.md cache split, implemented.

Stock run.py re-runs the whole decoder over the full padded buffer every token. Here:
  - encoder runs once (reuse the flax encoder),
  - cross-attention K/V are projected from the encoder output once per request,
  - decoder self-attention K/V are cached and appended,
  - a chunked `extend` processes m new tokens in parallel (m=1 for free decode; m=len(run)
    for a jump-forward forced run), writing their K/V and returning logits.

Weights are read straight from the released checkpoint's nn.scan-stacked pytree; the math
mirrors needle/model/architecture.py exactly (ZCRMSNorm, per-head q/k norm, GQA, half-split
RoPE on self-attn only, gated residuals, no FFN, tied embedding).
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp

from needle.model.run import load_checkpoint, _build_encoder_input
from needle.model.architecture import SimpleAttentionNetwork, make_padding_mask
from needle.dataset.dataset import get_tokenizer, DEFAULT_MAX_ENC_LEN

CKPT = "/storage/needle-e1/weights/needle.pkl"
F32 = jnp.float32


def _zcrms(x, scale, eps=1e-6):
    """ZCRMSNorm over the last axis: (1+scale) * x / rms(x). Compute in f32."""
    x = x.astype(F32)
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (1.0 + scale.astype(F32)) * x / rms


class KVNeedle:
    def __init__(self, max_dec=64, enc_max=640):
        self.max_dec = max_dec
        self.enc_max = enc_max
        params, cfg = load_checkpoint(CKPT)
        self.cfg = cfg
        self.model = SimpleAttentionNetwork(cfg)
        self.tok = get_tokenizer()
        self.params = params
        self.H = cfg.num_heads
        self.KV = cfg.num_kv_heads
        self.hd = cfg.d_model // cfg.num_heads
        self.L = cfg.num_decoder_layers
        self.rep = self.H // self.KV
        self.embed_scale = float(np.sqrt(cfg.d_model))

        # pull stacked decoder weights to f32 arrays: [layer] index on axis 0
        d = params["decoder"]["layers"]["DecoderBlock_0"]
        g = lambda *ks: jnp.asarray(_dig(d, ks), dtype=F32)
        self.sn0 = g("ZCRMSNorm_0", "scale")            # (L,512)  self pre-norm
        self.sn1 = g("ZCRMSNorm_1", "scale")            # (L,512)  cross pre-norm
        self.sWq = g("self_attn", "q_proj", "kernel")   # (L,512,512)
        self.sWk = g("self_attn", "k_proj", "kernel")   # (L,512,256)
        self.sWv = g("self_attn", "v_proj", "kernel")
        self.sWo = g("self_attn", "out_proj", "kernel")
        self.sqn = g("self_attn", "q_norm", "scale")    # (L,64)
        self.skn = g("self_attn", "k_norm", "scale")
        self.sgate = g("self_attn_gate")                # (L,)
        self.cWq = g("cross_attn", "q_proj", "kernel")
        self.cWk = g("cross_attn", "k_proj", "kernel")
        self.cWv = g("cross_attn", "v_proj", "kernel")
        self.cWo = g("cross_attn", "out_proj", "kernel")
        self.cqn = g("cross_attn", "q_norm", "scale")
        self.ckn = g("cross_attn", "k_norm", "scale")
        self.cgate = g("cross_attn_gate")
        self.final = jnp.asarray(params["decoder"]["ZCRMSNorm_0"]["scale"], F32)  # (512,)
        self.emb = jnp.asarray(params["embedding"]["embedding"], F32)             # (V,512)

        # RoPE tables (half-split): freqs over hd, positions 0..max_dec
        inv = 1.0 / (cfg.rope_theta ** (np.arange(0, self.hd, 2) / self.hd))
        t = np.arange(max_dec)
        ang = np.outer(t, inv)
        self.cos = jnp.asarray(np.cos(ang), F32)  # (max_dec, hd/2)
        self.sin = jnp.asarray(np.sin(ang), F32)

        self._extend_cache = {}

    # ---- encoder + cross-KV (prefill) ----
    def encode(self, query, tools_json, max_enc_len=DEFAULT_MAX_ENC_LEN):
        toks = _build_encoder_input(self.tok, query, tools_json, max_enc_len)
        enc_input = jnp.array([toks])
        src_mask = make_padding_mask(enc_input, self.tok.pad_token_id)
        enc_out, enc_mask = self.model.apply(
            {"params": self.params}, enc_input, src_mask=src_mask, method="encode")
        return enc_out, enc_mask, len(toks)

    def cross_kv(self, enc_out, enc_mask):
        """Project encoder output to per-layer cross K/V (padded to enc_max) + bias."""
        e = jnp.asarray(enc_out[0], F32)            # (enc,512)
        enc = e.shape[0]
        valid = np.asarray(enc_mask[0, 0, 0])       # (enc,) bool
        ck = jnp.einsum("ed,ldk->lek", e, self.cWk).reshape(self.L, enc, self.KV, self.hd)
        cv = jnp.einsum("ed,ldk->lek", e, self.cWv).reshape(self.L, enc, self.KV, self.hd)
        ck = _zcrms(ck, self.ckn[:, None, None, :])                 # k_norm
        ck = jnp.repeat(ck, self.rep, axis=2)                       # (L,enc,H,hd)
        cv = jnp.repeat(cv, self.rep, axis=2)
        bias = jnp.where(jnp.asarray(valid)[None, :], 0.0, -1e30)   # (1,enc)
        # pad to enc_max for jit shape stability
        pad = self.enc_max - enc
        ck = jnp.pad(ck, ((0, 0), (0, pad), (0, 0), (0, 0)))
        cv = jnp.pad(cv, ((0, 0), (0, pad), (0, 0), (0, 0)))
        bias = jnp.pad(bias, ((0, 0), (0, pad)), constant_values=-1e30)  # (1,enc_max)
        return ck, cv, bias[0]

    def fresh_cache(self):
        return (jnp.zeros((self.L, self.max_dec, self.KV, self.hd), F32),
                jnp.zeros((self.L, self.max_dec, self.KV, self.hd), F32))

    def _get_extend(self, m):
        if m not in self._extend_cache:
            self._extend_cache[m] = jax.jit(functools.partial(self._extend_impl, m))
        return self._extend_cache[m]

    def _extend_impl(self, m, token_ids, start, kc, vc, ck, cv, cbias):
        """Process m tokens at positions [start, start+m). Returns (logits(m,V), kc, vc)."""
        H, KV, hd, rep = self.H, self.KV, self.hd, self.rep
        pos = start + jnp.arange(m)                        # (m,) global positions
        cos = self.cos[pos]; sin = self.sin[pos]           # (m, hd/2)

        def rope(x):  # x: (m, nh, hd)
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            c = cos[:, None, :]; s = sin[:, None, :]
            return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)

        h = self.emb[token_ids] * self.embed_scale         # (m,512)
        kpos = jnp.arange(self.max_dec)                    # key slots
        allow_self = (kpos[None, :] <= pos[:, None])       # (m, max_dec)

        for l in range(self.L):
            # ---- self attention ----
            xn = _zcrms(h, self.sn0[l])
            q = (xn @ self.sWq[l]).reshape(m, H, hd)
            k = (xn @ self.sWk[l]).reshape(m, KV, hd)
            v = (xn @ self.sWv[l]).reshape(m, KV, hd)
            q = _zcrms(q, self.sqn[l]); k = _zcrms(k, self.skn[l])
            q = rope(q); k = rope(k)
            kc = jax.lax.dynamic_update_slice(kc, k[None], (l, start, 0, 0))
            vc = jax.lax.dynamic_update_slice(vc, v[None], (l, start, 0, 0))
            Kr = jnp.repeat(kc[l], rep, axis=1)            # (max_dec,H,hd)
            Vr = jnp.repeat(vc[l], rep, axis=1)
            sc = jnp.einsum("qhd,khd->hqk", q, Kr) / np.sqrt(hd)     # (H,m,max_dec)
            sc = jnp.where(allow_self[None], sc, -1e30)
            a = jax.nn.softmax(sc, axis=-1)
            o = jnp.einsum("hqk,khd->qhd", a, Vr).reshape(m, H * hd)
            o = o @ self.sWo[l]
            h = h + jax.nn.sigmoid(self.sgate[l]) * o
            # ---- cross attention ----
            xn = _zcrms(h, self.sn1[l])
            cq = _zcrms((xn @ self.cWq[l]).reshape(m, H, hd), self.cqn[l])  # no rope
            sc = jnp.einsum("qhd,khd->hqk", cq, ck[l]) / np.sqrt(hd)        # (H,m,enc_max)
            sc = sc + cbias[None, None, :]
            a = jax.nn.softmax(sc, axis=-1)
            o = jnp.einsum("hqk,khd->qhd", a, cv[l]).reshape(m, H * hd) @ self.cWo[l]
            h = h + jax.nn.sigmoid(self.cgate[l]) * o

        h = _zcrms(h, self.final)
        logits = h @ self.emb.T                            # (m, V)
        return logits, kc, vc

    def extend(self, token_ids, start, kc, vc, ck, cv, cbias):
        m = len(token_ids)
        fn = self._get_extend(m)
        toks = jnp.asarray(np.asarray(token_ids, np.int32))
        return fn(toks, jnp.int32(start), kc, vc, ck, cv, cbias)


def _dig(d, ks):
    for k in ks:
        d = d[k]
    return d
