"""Batched KV-cached Needle decoder — the GPU throughput lever.

`kv_model.KVNeedle` is batch-1: on a 4090 that leaves the 26M model latency-bound (a full
decode pass is ~0.6ms, so wall time is host/dispatch, not compute — see GPU_NOTES.md). The
win the batch-1 4-config benchmark *couldn't* show is running many independent requests through
the cache at once: each decode step becomes one batched extend with a single host argmax for
the whole batch, so per-stream throughput holds while aggregate throughput scales with the card.

This subclasses KVNeedle to reuse its weight load + RoPE tables, and adds a batch axis B to the
encoder, the cross-K/V projection, the self-attention cache, and the chunked `extend`. The math
is identical to the batch-1 path (and to needle/model/architecture.py) — just carrying B.

Cross-K/V are cached as KV-heads (repeated to H inside the kernel, as self-attn already does) so
the cross cache is num_heads/num_kv_heads x smaller — the difference between B=256 fitting or not.
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp

from needle.model.architecture import make_padding_mask
from needle.dataset.dataset import DEFAULT_MAX_ENC_LEN

from kv_model import KVNeedle, _zcrms, F32
from needle.model.run import _build_encoder_input


class BatchKVNeedle(KVNeedle):
    """KVNeedle with a leading batch axis on every decode-time tensor."""

    # ---- prefill ----
    def encode(self, queries, tools_json, max_enc_len=DEFAULT_MAX_ENC_LEN):
        """Batched encode. queries: list[str]. Returns enc_out (B,enc_max,d), mask, real_lens."""
        pad = self.tok.pad_token_id
        lists, real = [], []
        for q in queries:
            t = _build_encoder_input(self.tok, q, tools_json, max_enc_len)[:self.enc_max]
            real.append(len(t))
            lists.append(t)
        arr = np.full((len(queries), self.enc_max), pad, dtype=np.int32)
        for i, t in enumerate(lists):
            arr[i, :len(t)] = t
        enc_input = jnp.asarray(arr)
        src_mask = make_padding_mask(enc_input, pad)
        enc_out, enc_mask = self.model.apply(
            {"params": self.params}, enc_input, src_mask=src_mask, method="encode")
        return enc_out, enc_mask, real

    def cross_kv(self, enc_out, enc_mask):
        """Per-layer cross K/V, KV-heads (not repeated), + additive pad bias. Batched."""
        e = jnp.asarray(enc_out, F32)                       # (B, enc_max, d)
        B = e.shape[0]
        valid = np.asarray(enc_mask[:, 0, 0, :])            # (B, enc_max) bool
        ck = jnp.einsum("bed,ldk->lbek", e, self.cWk).reshape(self.L, B, self.enc_max, self.KV, self.hd)
        cv = jnp.einsum("bed,ldk->lbek", e, self.cWv).reshape(self.L, B, self.enc_max, self.KV, self.hd)
        ck = _zcrms(ck, self.ckn[:, None, None, None, :])   # k_norm over hd
        bias = jnp.where(jnp.asarray(valid), 0.0, -1e30)    # (B, enc_max)
        return ck, cv, bias

    def fresh_cache(self, B):
        z = jnp.zeros((self.L, B, self.max_dec, self.KV, self.hd), F32)
        return z, z

    # ---- chunked extend (batched) ----
    def _get_extend(self, key):
        if key not in self._extend_cache:
            self._extend_cache[key] = jax.jit(functools.partial(self._extend_impl_b, key))
        return self._extend_cache[key]

    def _extend_impl_b(self, m, token_ids, start, kc, vc, ck, cv, cbias):
        """token_ids (B,m) at positions [start,start+m). Returns logits (B,m,V), kc, vc."""
        H, KV, hd, rep = self.H, self.KV, self.hd, self.rep
        B = token_ids.shape[0]
        pos = start + jnp.arange(m)                         # (m,) uniform across batch
        cos = self.cos[pos]; sin = self.sin[pos]            # (m, hd/2)

        def rope(x):  # x: (B, m, nh, hd)
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            c = cos[None, :, None, :]; s = sin[None, :, None, :]
            return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)

        h = self.emb[token_ids] * self.embed_scale          # (B, m, d)
        kpos = jnp.arange(self.max_dec)
        allow_self = (kpos[None, :] <= pos[:, None])         # (m, max_dec)

        for l in range(self.L):
            # ---- self attention ----
            xn = _zcrms(h, self.sn0[l])
            q = (xn @ self.sWq[l]).reshape(B, m, H, hd)
            k = (xn @ self.sWk[l]).reshape(B, m, KV, hd)
            v = (xn @ self.sWv[l]).reshape(B, m, KV, hd)
            q = _zcrms(q, self.sqn[l]); k = _zcrms(k, self.skn[l])
            q = rope(q); k = rope(k)
            kc = jax.lax.dynamic_update_slice(kc, k[None], (l, 0, start, 0, 0))
            vc = jax.lax.dynamic_update_slice(vc, v[None], (l, 0, start, 0, 0))
            Kr = jnp.repeat(kc[l], rep, axis=2)             # (B, max_dec, H, hd)
            Vr = jnp.repeat(vc[l], rep, axis=2)
            sc = jnp.einsum("bqhd,bkhd->bhqk", q, Kr) / np.sqrt(hd)   # (B,H,m,max_dec)
            sc = jnp.where(allow_self[None, None], sc, -1e30)
            a = jax.nn.softmax(sc, axis=-1)
            o = jnp.einsum("bhqk,bkhd->bqhd", a, Vr).reshape(B, m, H * hd) @ self.sWo[l]
            h = h + jax.nn.sigmoid(self.sgate[l]) * o
            # ---- cross attention ----
            xn = _zcrms(h, self.sn1[l])
            cq = _zcrms((xn @ self.cWq[l]).reshape(B, m, H, hd), self.cqn[l])   # no rope
            ckl = jnp.repeat(ck[l], rep, axis=2)            # (B, enc_max, H, hd)
            cvl = jnp.repeat(cv[l], rep, axis=2)
            sc = jnp.einsum("bqhd,bkhd->bhqk", cq, ckl) / np.sqrt(hd)           # (B,H,m,enc_max)
            sc = sc + cbias[:, None, None, :]
            a = jax.nn.softmax(sc, axis=-1)
            o = jnp.einsum("bhqk,bkhd->bqhd", a, cvl).reshape(B, m, H * hd) @ self.cWo[l]
            h = h + jax.nn.sigmoid(self.cgate[l]) * o

        h = _zcrms(h, self.final)
        logits = h @ self.emb.T                             # (B, m, V)
        return logits, kc, vc

    def extend(self, token_ids, start, kc, vc, ck, cv, cbias):
        """token_ids: (B, m) int array. start: int."""
        toks = jnp.asarray(np.asarray(token_ids, np.int32))
        m = toks.shape[1]
        fn = self._get_extend(m)
        return fn(toks, jnp.int32(start), kc, vc, ck, cv, cbias)

    # ---- greedy driver ----
    def greedy(self, enc_out, enc_mask, max_new=48):
        """Batched greedy decode. Returns list[str] (one per row), stripped of <tool_call>."""
        B = enc_out.shape[0]
        eos = self.tok.eos_token_id
        ck, cv, cbias = self.cross_kv(enc_out, enc_mask)
        kc, vc = self.fresh_cache(B)
        toks = np.full((B, 1), eos, dtype=np.int32)
        logits, kc, vc = self.extend(toks, 0, kc, vc, ck, cv, cbias)
        gen = [[] for _ in range(B)]
        done = np.zeros(B, dtype=bool)
        for pos in range(1, max_new):
            nt = np.asarray(jnp.argmax(logits[:, -1], axis=-1))   # (B,) — one host sync/step
            for i in range(B):
                if done[i]:
                    continue
                if nt[i] == eos:
                    done[i] = True
                else:
                    gen[i].append(int(nt[i]))
            if done.all():
                break
            logits, kc, vc = self.extend(nt[:, None], pos, kc, vc, ck, cv, cbias)
        out = []
        for row in gen:
            t = self.tok.decode(row)
            out.append(t[len("<tool_call>"):].strip() if t.startswith("<tool_call>") else t.strip())
        return out
