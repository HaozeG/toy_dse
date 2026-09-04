from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import Qwen3Config

CONFIG_PATH = Path(__file__).parent / "qwen3_0.6b.config.json"

EXPECTED = {
    "num_hidden_layers": 28,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}

ELEM_BYTES = 1  # int8 weights and activations; simulator uses width only


@dataclass(frozen=True)
class DecodeOp:
    name: str
    macs: int
    weight_bytes: int
    kv_bytes: int
    activation_bytes: int
    vector_ops: int = 0


def load_qwen3_0_6b() -> Qwen3Config:
    data = json.loads(CONFIG_PATH.read_text())
    cfg = Qwen3Config.from_dict(data)
    for key, expected in EXPECTED.items():
        got = getattr(cfg, key)
        if got != expected:
            raise AssertionError(f"Qwen3-0.6B config {key}: expected {expected}, got {got}")
    return cfg


def build_decode_ops(
    cfg: Qwen3Config | None = None,
    *,
    kv_len: int,
    batch: int,
    elem_bytes: int = ELEM_BYTES,
) -> list[DecodeOp]:
    """Per-token decode op graph. Weights are streamed once (reused across batch)."""
    if cfg is None:
        cfg = load_qwen3_0_6b()
    b = batch
    k = kv_len
    h = cfg.hidden_size
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d = cfg.head_dim
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size
    layers = cfg.num_hidden_layers
    q_dim = n_q * d
    kv_dim = n_kv * d
    e = elem_bytes

    ops: list[DecodeOp] = []
    ops.append(
        DecodeOp(
            name="embed_lookup",
            macs=0,
            weight_bytes=0,
            kv_bytes=0,
            activation_bytes=b * h * e,
        )
    )
    for layer in range(layers):
        prefix = f"L{layer}"
        ops.append(
            DecodeOp(
                name=f"{prefix}.attn_norm",
                macs=0,
                weight_bytes=h * e,
                kv_bytes=0,
                activation_bytes=b * h * e,
                vector_ops=b * h,
            )
        )
        ops.append(DecodeOp(f"{prefix}.q_proj", b * h * q_dim, h * q_dim * e, 0, b * q_dim * e))
        ops.append(DecodeOp(f"{prefix}.k_proj", b * h * kv_dim, h * kv_dim * e, 0, b * kv_dim * e))
        ops.append(DecodeOp(f"{prefix}.v_proj", b * h * kv_dim, h * kv_dim * e, 0, b * kv_dim * e))
        ops.append(
            DecodeOp(
                name=f"{prefix}.q_norm",
                macs=0,
                weight_bytes=d * e,
                kv_bytes=0,
                activation_bytes=b * q_dim * e,
                vector_ops=b * q_dim,
            )
        )
        ops.append(
            DecodeOp(
                name=f"{prefix}.k_norm",
                macs=0,
                weight_bytes=d * e,
                kv_bytes=0,
                activation_bytes=b * kv_dim * e,
                vector_ops=b * kv_dim,
            )
        )
        ops.append(
            DecodeOp(
                name=f"{prefix}.rope",
                macs=0,
                weight_bytes=0,
                kv_bytes=0,
                activation_bytes=b * (q_dim + kv_dim) * e,
                vector_ops=b * (q_dim + kv_dim),
            )
        )
        ops.append(
            DecodeOp(
                name=f"{prefix}.attn_qk",
                macs=b * n_q * k * d,
                weight_bytes=0,
                kv_bytes=b * k * kv_dim * e,
                activation_bytes=b * n_q * k * e,
            )
        )
        ops.append(
            DecodeOp(
                name=f"{prefix}.attn_av",
                macs=b * n_q * k * d,
                weight_bytes=0,
                kv_bytes=b * k * kv_dim * e,
                activation_bytes=b * q_dim * e,
            )
        )
        ops.append(DecodeOp(f"{prefix}.o_proj", b * q_dim * h, q_dim * h * e, 0, b * h * e))
        ops.append(
            DecodeOp(
                name=f"{prefix}.mlp_norm",
                macs=0,
                weight_bytes=h * e,
                kv_bytes=0,
                activation_bytes=b * h * e,
                vector_ops=b * h,
            )
        )
        ops.append(DecodeOp(f"{prefix}.gate_proj", b * h * inter, h * inter * e, 0, b * inter * e))
        ops.append(DecodeOp(f"{prefix}.up_proj", b * h * inter, h * inter * e, 0, b * inter * e))
        ops.append(
            DecodeOp(
                name=f"{prefix}.silu_mul",
                macs=0,
                weight_bytes=0,
                kv_bytes=0,
                activation_bytes=b * inter * e,
                vector_ops=b * inter,
            )
        )
        ops.append(DecodeOp(f"{prefix}.down_proj", b * inter * h, inter * h * e, 0, b * h * e))
    ops.append(
        DecodeOp(
            name="final_norm",
            macs=0,
            weight_bytes=h * e,
            kv_bytes=0,
            activation_bytes=b * h * e,
            vector_ops=b * h,
        )
    )
    # Tied embeddings: stream the table once as lm_head weights.
    ops.append(DecodeOp("lm_head", b * h * vocab, vocab * h * e, 0, b * vocab * e))
    return ops


def summarize_ops(ops: list[DecodeOp]) -> dict[str, Any]:
    return {
        "macs": sum(op.macs for op in ops),
        "weight_bytes": sum(op.weight_bytes for op in ops),
        "kv_bytes": sum(op.kv_bytes for op in ops),
        "activation_bytes": sum(op.activation_bytes for op in ops),
        "vector_ops": sum(op.vector_ops for op in ops),
        "n_ops": len(ops),
    }
