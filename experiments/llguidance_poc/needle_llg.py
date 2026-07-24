"""Needle decode drivers: unconstrained (stock) vs llguidance-grammar + jump-forward.

Both arms share the same jitted full-buffer decode step (the stock un-cached path, so
wall-clock is ~proportional to the number of forward passes). The constrained arm masks
logits with an llguidance matcher and injects grammar-forced token runs in a single pass
(jump-forward), so it issues far fewer forward passes on Needle's highly-scaffolded output.
"""
import os, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp
import numpy as np
import sentencepiece as spm

from llguidance import LLTokenizer, TokenizerWrapper, LLMatcher
from llguidance.numpy import allocate_token_bitmask, fill_next_token_bitmask

from needle.model.run import load_checkpoint, _build_encoder_input
from needle.model.architecture import SimpleAttentionNetwork, make_causal_mask, make_padding_mask
from needle.dataset.dataset import get_tokenizer, DEFAULT_MAX_ENC_LEN

from tools_spec import build_union_grammar

CKPT = "/storage/needle-e1/weights/needle.pkl"
SP_PATH = "/storage/needle-e1/weights/tokenizer/needle.model"
# Fixed encoder length so the full-buffer decode_step compiles ONCE. The encoder input
# is variable-length; padding it (pad positions are masked in cross-attn) keeps the output
# identical but pins encoder_out's shape, avoiding a per-prompt XLA recompile of _decode —
# invisible on CPU but a ~1.8s/prompt tax on GPU. Matches KVNeedle's enc_max.
ENC_PAD = 640


def _token_bytes(sp, i):
    if sp.IsControl(i) or sp.IsUnknown(i):
        return b""
    if sp.IsByte(i):
        p = sp.IdToPiece(i)
        try:
            return bytes([int(p[1:-1], 16)]) if p.startswith("<0x") else p.encode()
        except Exception:
            return b""
    return sp.IdToPiece(i).replace("▁", " ").encode("utf-8")


class NeedleLLG:
    def __init__(self, max_gen=48):
        self.max_gen = max_gen
        self.params, self.config = load_checkpoint(CKPT)
        self.model = SimpleAttentionNetwork(self.config)
        self.tok = get_tokenizer()
        self.pad_id = self.tok.pad_token_id
        self.eos_id = self.tok.eos_token_id

        # llguidance tokenizer bridge over Needle SentencePiece
        sp = spm.SentencePieceProcessor(); sp.Load(SP_PATH)
        self.sp = sp
        self.V = sp.GetPieceSize()
        gtok = _make_gtok(sp, self.V, self.eos_id)
        self.lltok = LLTokenizer(TokenizerWrapper(gtok))
        self.grammar = LLMatcher.grammar_from_lark(build_union_grammar())
        warn = LLMatcher.validate_grammar(self.grammar, self.lltok)
        if warn:
            raise RuntimeError("grammar invalid: " + warn)

        # Invariant envelope prefix ' [{"name":"'. The model emits it identically on
        # every call as the SentencePiece dummy-prefixed run [356,294,264]; after it,
        # the stream is pure continuation (no dummy prefix). We inject it as a known
        # forced run (== jump-forward) so the ff/mask loop only runs from the tool name
        # onward, where the no-dummy-prefix encoder matches the model.
        self.prefix_tokens = self.sp.EncodeAsIds(' [{"name":"')
        _m = LLMatcher(self.lltok, self.grammar)
        if not _m.try_consume_tokens(self.prefix_tokens):
            raise RuntimeError(f"prefix tokens {self.prefix_tokens} rejected by grammar")

        tgt_mask = make_causal_mask(max_gen)

        @jax.jit
        def decode_step(params, dec_buffer, encoder_out, cross_mask):
            return self.model.apply({"params": params}, dec_buffer, encoder_out,
                                    self_mask=tgt_mask, cross_mask=cross_mask, method="decode")
        self._decode_step = decode_step

    # ---- encoder (shared by both arms) ----
    def encode(self, query, tools_json):
        enc_tokens = _build_encoder_input(self.tok, query, tools_json, DEFAULT_MAX_ENC_LEN)
        enc_tokens = enc_tokens[:ENC_PAD] + [self.pad_id] * max(0, ENC_PAD - len(enc_tokens))
        enc_input = jnp.array([enc_tokens])
        src_mask = make_padding_mask(enc_input, self.pad_id)
        encoder_out, enc_mask = self.model.apply(
            {"params": self.params}, enc_input, src_mask=src_mask, method="encode")
        return encoder_out, enc_mask

    def _fresh_buffer(self):
        buf = np.full((1, self.max_gen), self.pad_id, dtype=np.int32)
        buf[0, 0] = self.eos_id
        return buf

    def _decode(self, buf, enc_out, enc_mask):
        return self._decode_step(self.params, jnp.asarray(buf), enc_out, enc_mask)

    def _finish(self, gen):
        text = self.tok.decode(gen)
        if text.startswith("<tool_call>"):
            text = text[len("<tool_call>"):]
        return text.strip()

    # ---- arm A: stock unconstrained greedy ----
    def gen_unconstrained(self, enc_out, enc_mask):
        buf = self._fresh_buffer()
        t0 = time.perf_counter()
        logits = self._decode(buf, enc_out, enc_mask); passes = 1
        gen = []
        for i in range(self.max_gen - 1):
            t = int(jnp.argmax(logits[0, i]))
            if t == self.eos_id:
                break
            gen.append(t); buf[0, i + 1] = t
            logits = self._decode(buf, enc_out, enc_mask); passes += 1
        dt = (time.perf_counter() - t0) * 1000
        return {"text": self._finish(gen), "passes": passes, "tokens": len(gen), "ms": dt}

    # ---- arm B: llguidance grammar + jump-forward ----
    def gen_constrained(self, enc_out, enc_mask):
        buf = self._fresh_buffer()
        mask_buf = allocate_token_bitmask(1, self.lltok.vocab_size)
        t0 = time.perf_counter()
        logits = self._decode(buf, enc_out, enc_mask); passes = 1
        # position 0: free (model emits <tool_call>); grammar covers only the JSON after it
        t0tok = int(jnp.argmax(logits[0, 0]))
        buf[0, 1] = t0tok
        gen = [t0tok]
        i = 1
        m = LLMatcher(self.lltok, self.grammar)
        # inject the invariant ' [{"name":"' prefix as a forced run (jump-forward)
        m.consume_tokens(self.prefix_tokens)
        for k, t in enumerate(self.prefix_tokens):
            if i + 1 + k < self.max_gen:
                buf[0, i + 1 + k] = t
        pf = min(len(self.prefix_tokens), self.max_gen - 1 - i)
        gen.extend(self.prefix_tokens[:pf]); i += pf
        ff_total = pf
        logits = self._decode(buf, enc_out, enc_mask); passes += 1
        while i < self.max_gen - 1:
            ff = m.compute_ff_tokens()
            if ff:
                run, stop = [], False
                for t in ff:
                    if t == self.eos_id:
                        stop = True; break
                    run.append(t)
                if run:
                    m.consume_tokens(run)
                    for k, t in enumerate(run):
                        if i + 1 + k < self.max_gen:
                            buf[0, i + 1 + k] = t
                    fit = min(len(run), self.max_gen - 1 - i)
                    gen.extend(run[:fit]); i += fit
                    ff_total += fit
                if stop or m.is_stopped() or i >= self.max_gen - 1:
                    break
                logits = self._decode(buf, enc_out, enc_mask); passes += 1
                continue
            if m.is_stopped():
                break
            fill_next_token_bitmask(m, mask_buf, 0)
            allowed = np.unpackbits(mask_buf.view(np.uint8), bitorder="little")[:self.V]
            row = np.array(logits[0, i], dtype=np.float32)  # writable copy
            row[allowed == 0] = -np.inf
            if not np.isfinite(row).any():
                break
            t = int(np.argmax(row))
            m.consume_token(t)
            buf[0, i + 1] = t; gen.append(t); i += 1
            if m.is_stopped():
                break
            logits = self._decode(buf, enc_out, enc_mask); passes += 1
        jax.block_until_ready(logits)  # forced-run passes dispatch without a read; await them
        dt = (time.perf_counter() - t0) * 1000
        return {"text": self._finish(gen), "passes": passes, "tokens": len(gen),
                "ms": dt, "forced": ff_total}


def _no_dummy_prefix_sp(sp_path):
    """A SentencePiece processor with add_dummy_prefix disabled.

    Needle emits JSON mid-stream with no word-boundary space, so it never re-adds
    SentencePiece's leading '▁'. llguidance tokenizes grammar-forced byte runs via this
    encoder; with the dummy prefix on, a spurious '▁' (space) leaks into forced tool
    names (e.g. 'rate _product'). Disabling it makes forced tokenization match the model.
    """
    from sentencepiece import sentencepiece_model_pb2 as pb
    proto = pb.ModelProto()
    proto.ParseFromString(open(sp_path, "rb").read())
    proto.normalizer_spec.add_dummy_prefix = False
    sp2 = spm.SentencePieceProcessor()
    sp2.LoadFromSerializedProto(proto.SerializeToString())
    return sp2


class _CallableG:
    """Minimal gtokenizer TokenizerWrapper expects: byte-per-token vocab + encoder."""
    def __init__(self, sp, V, eos_id, sp_path):
        self.sp_enc = _no_dummy_prefix_sp(sp_path)
        self.eos_token_id = eos_id
        self.bos_token_id = None
        self.tokens = [_token_bytes(sp, i) for i in range(V)]
        self.special_token_ids = [0, 1, 2, 3]  # pad, eos, bos, unk; tool_call/tools stay content

    def __call__(self, s):
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="ignore")
        return self.sp_enc.EncodeAsIds(s)


def _make_gtok(sp, V, eos_id, sp_path=SP_PATH):
    return _CallableG(sp, V, eos_id, sp_path)
