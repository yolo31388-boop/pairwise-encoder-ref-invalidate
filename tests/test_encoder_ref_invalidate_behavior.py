import media.encoder_ref_invalidate as mod
from media.encoder_ref_invalidate import EncoderState


def _p(gen, ref_ids, type_="P"):
    return {"type": type_, "ref_ids": list(ref_ids), "ref_generation": gen}


# 1. IDR 时清空参考列表
def test_idr_clears_reference_list():
    s = EncoderState()
    mod.idr_request(s)
    mod.encode(s, {"type": "IDR"})
    mod.encode(s, _p(s.generation, [s.refs[-1].frame_id]))
    existing = len(s.refs)
    assert existing == 2
    res = mod.idr_request(s)
    assert res["cleared"] == existing
    assert s.refs == []
    assert res["refs"] == []


def test_idr_anchor_belongs_to_new_generation():
    s = EncoderState()
    mod.idr_request(s)
    out = mod.encode(s, {"type": "IDR"})
    assert out["status"] == "encoded"
    assert all(r.generation == s.generation for r in s.refs)


# 2. 作废队列内在途帧
def test_idr_drops_in_flight_frames_with_stale_generation():
    s = EncoderState()
    mod.idr_request(s)
    mod.encode(s, {"type": "IDR"})
    anchor = s.refs[-1].frame_id
    old_gen = s.generation

    from media.encoder_ref_invalidate import QueuedFrame

    s.queue.append(QueuedFrame(100, "P", (anchor,), old_gen))
    s.queue.append(QueuedFrame(101, "B", (), old_gen))
    res = mod.idr_request(s)

    assert 100 in res["dropped"] or 101 in res["dropped"]
    assert all(f.ref_generation == s.generation for f in s.queue)
    assert s.counters["dropped"] >= 1


# 3. 重置码率模型
def test_idr_resets_rate_model():
    s = EncoderState()
    mod.idr_request(s)
    mod.encode(s, {"type": "IDR"})
    for _ in range(10):
        mod.encode(s, _p(s.generation, [s.refs[-1].frame_id]))
    s.complexity = 3.5
    s.bits_used = 12_000

    res = mod.idr_request(s)
    assert s.complexity == s.base_complexity
    assert s.bits_used == 0
    assert s.gop_index == 0
    assert s.target_bits == s.idr_target_bits
    assert res["rate"]["after"]["complexity"] == s.base_complexity


def test_first_frame_after_switch_does_not_overshoot():
    s = EncoderState()
    mod.idr_request(s)
    mod.encode(s, {"type": "IDR"})
    for _ in range(10):
        mod.encode(s, _p(s.generation, [s.refs[-1].frame_id]))
    # 模拟旧 GOP 累积的高复杂度
    s.complexity = 4.0

    mod.idr_request(s)
    mod.encode(s, {"type": "IDR"})
    out = mod.encode(s, _p(s.generation, [s.refs[-1].frame_id]))
    assert out["status"] == "encoded"
    assert out["bits"] <= s.target_bits


# 4. 复位重排序深度
def test_idr_resets_reorder_depth_and_buffer():
    s = EncoderState()
    mod.idr_request(s)
    s.reorder_depth = 5
    s.reorder_buffer = [7, 8, 9]
    res = mod.idr_request(s)
    assert s.reorder_depth == s.base_reorder_depth
    assert s.reorder_buffer == []
    assert res["reorder"]["flushed"] == [7, 8, 9]


def test_bframe_order_uses_reset_depth():
    s = EncoderState()
    mod.idr_request(s)
    plan = mod.bframe_order(s, 9)
    assert plan["reorder_depth"] == s.base_reorder_depth
    assert len(plan["encode_order"]) == len(plan["display_order"])
    if s.base_reorder_depth == 0:
        assert plan["encode_order"] == plan["display_order"]
    else:
        anchors = [i for i in plan["display_order"] if i % (s.base_reorder_depth + 1) == 0]
        assert plan["encode_order"][: len(anchors)] == anchors


# 5. 重传走新参考
def test_retransmit_after_idr_requires_new_reference():
    s = EncoderState()
    mod.idr_request(s)
    mod.encode(s, {"type": "IDR"})
    old_anchor = s.refs[-1].frame_id
    mod.idr_request(s)  # 清空参考

    res = mod.retransmit_ref(s, {"ref_id": old_anchor})
    assert res["requires_idr"] is True
    assert res["reference"] is None

    mod.encode(s, {"type": "IDR"})
    new_anchor = s.refs[-1].frame_id
    res2 = mod.retransmit_ref(s, {"ref_id": old_anchor})
    assert res2["requires_idr"] is False
    assert res2["reference"] == new_anchor
    assert res2["ignored_old_ref"] is True


# 6. 编码前校验参考代际
def test_encode_rejects_stale_generation():
    s = EncoderState()
    mod.idr_request(s)
    mod.encode(s, {"type": "IDR"})
    anchor = s.refs[-1].frame_id
    old_gen = s.generation
    mod.idr_request(s)

    res = mod.encode(s, _p(old_gen, [anchor]))
    assert res["status"] == "rejected"
    assert res["reason"] in ("stale_generation", "invalid_reference")
    assert res["fallback"]["action"] == "force_idr"
    assert s.counters["rejected"] == 1


def test_encode_accepts_current_generation_reference():
    s = EncoderState()
    mod.idr_request(s)
    mod.encode(s, {"type": "IDR"})
    out = mod.encode(s, _p(s.generation, [s.refs[-1].frame_id]))
    assert out["status"] == "encoded"
    assert out["generation"] == s.generation


# 辅助：校验 / 审计 / 确定性
def test_validation_flags_stale_ref():
    s = EncoderState()
    mod.idr_request(s)
    mod.encode(s, {"type": "IDR"})
    from media.encoder_ref_invalidate import RefFrame

    s.refs.append(RefFrame(999, s.generation - 1))
    assert mod.validation(s)["ok"] is False
    assert any(v["kind"] == "stale_ref" for v in mod.validation(s)["violations"])


def test_check_ok_after_clean_switch():
    s = EncoderState()
    mod.idr_request(s)
    mod.encode(s, {"type": "IDR"})
    report = mod.check(s)
    assert report["ok"] is True
    assert report["validation"]["violations"] == []


def test_audit_records_idr_switch():
    s = EncoderState()
    mod.idr_request(s)
    data = mod.audit(s)
    kinds = {e["kind"] for e in data["events"]}
    assert {"clear_refs", "invalidate_queue", "reset_rate", "reset_reorder", "idr_request"} <= kinds


def test_determinism_is_repeatable():
    res = mod.determinism()
    assert res["deterministic"] is True
    assert res == mod.determinism()
