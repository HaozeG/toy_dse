from hw.schema import load_hw
from sw.roofline import roofline_table
from sw.workload.qwen3 import build_decode_ops, load_qwen3_0_6b, summarize_ops


def test_qwen3_config_asserts():
    cfg = load_qwen3_0_6b()
    assert cfg.num_hidden_layers == 28
    assert cfg.hidden_size == 1024
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.intermediate_size == 3072
    assert cfg.vocab_size == 151936
    assert cfg.tie_word_embeddings is True


def test_grid8x8_b1_memory_bound():
    hw = load_hw("grid8x8")
    rows = { (r.kv_len, r.batch): r for r in roofline_table(hw) }
    for k in (128, 256, 1024):
        row = rows[(k, 1)]
        assert row.bound == "memory"
        assert row.mem_over_compute >= 50


def test_b16_ratio_drops_with_batch():
    hw = load_hw("grid8x8")
    rows = { (r.kv_len, r.batch): r for r in roofline_table(hw) }
    for k in (128, 256, 1024):
        r1 = rows[(k, 1)]
        r16 = rows[(k, 16)]
        assert r16.bound == "memory"
        assert r16.macs == r1.macs * 16
        assert r16.weight_bytes == r1.weight_bytes
        assert r16.mem_over_compute < r1.mem_over_compute
        # Weight-dominated K: intensity scales with B, so mem/cmp drops by ~B.
        # At K=1024 KV bytes also scale with B and the drop is smaller.
        if k <= 256:
            assert r16.mem_over_compute < r1.mem_over_compute / 8


def test_binding_named_for_all_points():
    hw = load_hw("grid8x8")
    rows = roofline_table(hw)
    assert len(rows) == 9
    assert all(r.bound in {"memory", "compute"} for r in rows)


def test_macs_scale_and_weights_independent_of_k():
    cfg = load_qwen3_0_6b()
    a = summarize_ops(build_decode_ops(cfg, kv_len=128, batch=1))
    b = summarize_ops(build_decode_ops(cfg, kv_len=256, batch=1))
    assert a["weight_bytes"] == b["weight_bytes"]
    assert b["kv_bytes"] == a["kv_bytes"] * 2
    assert b["macs"] > a["macs"]
