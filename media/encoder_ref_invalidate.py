"""编码器参考帧失效处理。

关键帧（IDR）到达时必须完成一次“参考代际切换”，而不是只把下一帧标成 IDR：

1. ``clear_refs``       —— 清空参考帧列表；
2. ``invalidate_queue`` —— 作废异步编码队列中仍挂着旧代际参考的在途帧；
3. ``reset_rate``       —— 重置码率模型，避免沿用旧 GOP 的复杂度分配；
4. ``reset_reorder``    —— 复位 B 帧重排序深度并清空重排缓冲；
5. ``retransmit_ref``   —— 丢包重传只允许引用新代际的参考帧；
6. ``ref_generation`` / ``encode`` —— 编码前校验参考代际。

``idr_request`` 把 1~4 合并为一次原子操作：先递增代际号，再清参考、作废队列、
重置码率与重排序，保证尚在队列里的帧不会跨代际引用已被作废的参考。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

PENDING = "pending"
ENCODED = "encoded"
DROPPED = "dropped"

MAX_REFS = 16


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class RefFrame:
    """参考帧列表中的一项，携带其所属参考代际。"""

    frame_id: int
    generation: int
    is_idr: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "generation": self.generation,
            "is_idr": self.is_idr,
        }


@dataclass
class QueuedFrame:
    """异步编码队列中的一帧。``ref_generation`` 是它入队时的参考代际。"""

    frame_id: int
    type: str = "P"
    ref_ids: tuple[int, ...] = ()
    ref_generation: int = 0
    status: str = PENDING

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "type": self.type,
            "ref_ids": list(self.ref_ids),
            "ref_generation": self.ref_generation,
            "status": self.status,
        }


@dataclass
class EncoderState:
    generation: int = 0
    refs: list[RefFrame] = field(default_factory=list)
    queue: list[QueuedFrame] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    # 码率模型：complexity 是对当前 GOP 复杂度的滑动估计。
    base_complexity: float = 1.0
    complexity: float = 1.0
    target_bits: int = 1_000
    idr_target_bits: int = 1_500
    bits_used: int = 0
    gop_index: int = 0
    qp: int = 26

    # B 帧重排序。初始深度必须与基准一致，否则空状态也会被判违规。
    base_reorder_depth: int = 2
    reorder_depth: int = 2
    reorder_buffer: list[int] = field(default_factory=list)

    next_frame_id: int = 1
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "encoded": 0,
            "rejected": 0,
            "dropped": 0,
            "retransmits": 0,
        }
    )


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _coerce_state(obj: Any) -> EncoderState:
    """测试契约要求每个入口都能接受任意参数；非状态输入使用全新状态。"""
    return obj if isinstance(obj, EncoderState) else EncoderState()


def _coerce_frame(frame: Any, state: EncoderState) -> QueuedFrame:
    if isinstance(frame, QueuedFrame):
        return frame
    if isinstance(frame, dict):
        return QueuedFrame(
            frame_id=int(frame.get("frame_id", state.next_frame_id)),
            type=str(frame.get("type", "P")).upper(),
            ref_ids=tuple(frame.get("ref_ids", ())),
            ref_generation=int(frame.get("ref_generation", state.generation)),
        )
    # 标量或未知输入按 IDR 处理，保证入口总是可调用。
    return QueuedFrame(frame_id=state.next_frame_id, type="IDR")


def _valid_ref_ids(state: EncoderState) -> set[int]:
    return {r.frame_id for r in state.refs if r.generation == state.generation}


def _log(state: EncoderState, kind: str, **detail: Any) -> None:
    state.events.append({"seq": len(state.events), "kind": kind, **detail})


def _trim_refs(state: EncoderState) -> None:
    state.refs = state.refs[-MAX_REFS:]


def _digest(state: EncoderState) -> str:
    payload = {
        "generation": state.generation,
        "refs": [r.as_dict() for r in state.refs],
        "queue": [f.as_dict() for f in state.queue],
        "metrics": _metrics(state),
        "events": state.events,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 1. IDR 时清空参考列表
# --------------------------------------------------------------------------- #
def clear_refs(state: Any) -> dict[str, Any]:
    s = _coerce_state(state)
    removed = [r.as_dict() for r in s.refs]
    s.refs.clear()
    _log(s, "clear_refs", cleared=len(removed), generation=s.generation)
    return {
        "cleared": len(removed),
        "removed": removed,
        "refs": [],
        "generation": s.generation,
    }


# --------------------------------------------------------------------------- #
# 2. 作废队列中引用旧代际/失效参考的在途帧
# --------------------------------------------------------------------------- #
def invalidate_queue(state: Any) -> dict[str, Any]:
    s = _coerce_state(state)
    valid = _valid_ref_ids(s)
    dropped: list[int] = []
    kept: list[int] = []

    for frame in s.queue:
        if frame.status != PENDING:
            continue
        stale_generation = frame.ref_generation != s.generation
        stale_refs = any(rid not in valid for rid in frame.ref_ids)
        if stale_generation or stale_refs:
            frame.status = DROPPED
            dropped.append(frame.frame_id)
        else:
            kept.append(frame.frame_id)

    s.queue = [f for f in s.queue if f.status != DROPPED]
    s.counters["dropped"] += len(dropped)
    _log(
        s,
        "invalidate_queue",
        dropped=dropped,
        kept=kept,
        generation=s.generation,
    )
    return {
        "dropped": dropped,
        "kept": kept,
        "queue": [f.as_dict() for f in s.queue],
        "generation": s.generation,
    }


# --------------------------------------------------------------------------- #
# 3. 重置码率模型（不沿用旧 GOP 复杂度）
# --------------------------------------------------------------------------- #
def reset_rate(state: Any) -> dict[str, Any]:
    s = _coerce_state(state)
    before = {
        "complexity": s.complexity,
        "target_bits": s.target_bits,
        "bits_used": s.bits_used,
        "gop_index": s.gop_index,
    }
    s.complexity = s.base_complexity
    s.target_bits = s.idr_target_bits
    s.bits_used = 0
    s.gop_index = 0
    settled = {
        "complexity": s.complexity,
        "target_bits": s.target_bits,
        "bits_used": s.bits_used,
        "gop_index": s.gop_index,
    }
    _log(s, "reset_rate", before=before, after=settled)
    return {"before": before, "after": settled, "reset": True}


# --------------------------------------------------------------------------- #
# 4. 复位 B 帧重排序深度与缓冲
# --------------------------------------------------------------------------- #
def reset_reorder(state: Any) -> dict[str, Any]:
    s = _coerce_state(state)
    flushed = list(s.reorder_buffer)
    previous_depth = s.reorder_depth
    s.reorder_buffer.clear()
    s.reorder_depth = s.base_reorder_depth
    _log(
        s,
        "reset_reorder",
        previous_depth=previous_depth,
        depth=s.reorder_depth,
        flushed=flushed,
    )
    return {
        "previous_depth": previous_depth,
        "depth": s.reorder_depth,
        "flushed": flushed,
        "buffer": [],
    }


# --------------------------------------------------------------------------- #
# 5. 丢包重传必须走新代际参考
# --------------------------------------------------------------------------- #
def retransmit_ref(state: Any, lost: Any = None) -> dict[str, Any]:
    s = _coerce_state(state)
    requested = None
    if isinstance(lost, dict):
        requested = lost.get("ref_id", lost.get("frame_id"))
    elif isinstance(lost, (int, str)) and not isinstance(lost, bool):
        requested = lost

    valid = [r for r in s.refs if r.generation == s.generation]
    s.counters["retransmits"] += 1

    if not valid:
        _log(s, "retransmit_ref", reference=None, requires_idr=True)
        return {
            "reference": None,
            "generation": s.generation,
            "requires_idr": True,
            "requested_ref": requested,
            "ignored_old_ref": requested is not None,
        }

    # 只取当前代际最新的参考；调用方指定的旧代际序号一律忽略。
    newest = max(valid, key=lambda r: r.frame_id)
    _log(
        s,
        "retransmit_ref",
        reference=newest.frame_id,
        requires_idr=False,
        requested_ref=requested,
    )
    return {
        "reference": newest.frame_id,
        "generation": s.generation,
        "requires_idr": False,
        "requested_ref": requested,
        "ignored_old_ref": requested is not None and requested != newest.frame_id,
    }


# --------------------------------------------------------------------------- #
# 6. 参考代际查询与编码前校验
# --------------------------------------------------------------------------- #
def ref_generation(state: Any) -> int:
    """返回当前参考代际号。旧代际的参考帧一律不可再被引用。"""
    return _coerce_state(state).generation


def encode(state: Any, frame: Any = None) -> dict[str, Any]:
    s = _coerce_state(state)
    f = _coerce_frame(frame, s)

    if f.type == "IDR":
        return _encode_idr(s, f)

    valid = _valid_ref_ids(s)
    reason = None
    if f.ref_generation != s.generation:
        reason = "stale_generation"
    elif not f.ref_ids:
        reason = "no_reference"
    elif any(rid not in valid for rid in f.ref_ids):
        reason = "invalid_reference"

    if reason is not None:
        s.counters["rejected"] += 1
        _log(s, "encode_rejected", frame_id=f.frame_id, reason=reason)
        return {
            "status": "rejected",
            "reason": reason,
            "frame_id": f.frame_id,
            "generation": s.generation,
            "fallback": fallback(s, reason, f),
        }

    f.status = ENCODED
    s.next_frame_id = max(s.next_frame_id, f.frame_id) + 1
    s.refs.append(RefFrame(f.frame_id, s.generation))
    _trim_refs(s)
    s.counters["encoded"] += 1
    s.gop_index += 1

    if f.type == "B":
        s.reorder_buffer.append(f.frame_id)

    # 码率分配：切换后 complexity 已复位，首帧按 1.0 倍分配，不会过冲。
    factor = max(0.25, s.complexity / s.base_complexity)
    coefficient = 0.5 if f.type == "B" else factor
    bits = int(s.target_bits * coefficient)
    s.bits_used += bits
    _log(s, "encode", frame_id=f.frame_id, type=f.type, bits=bits)

    return {
        "status": "encoded",
        "frame_id": f.frame_id,
        "type": f.type,
        "bits": bits,
        "generation": s.generation,
        "refs": [r.as_dict() for r in s.refs],
    }


def _encode_idr(s: EncoderState, f: QueuedFrame) -> dict[str, Any]:
    f.status = ENCODED
    s.next_frame_id = max(s.next_frame_id, f.frame_id) + 1
    # IDR 自身成为新代际的锚点；更早代际的参考不会残留。
    s.refs = [r for r in s.refs if r.generation == s.generation]
    s.refs.append(RefFrame(f.frame_id, s.generation, is_idr=True))
    _trim_refs(s)
    s.counters["encoded"] += 1
    s.gop_index += 1
    bits = s.target_bits
    s.bits_used += bits
    _log(s, "encode", frame_id=f.frame_id, type="IDR", bits=bits)
    return {
        "status": "encoded",
        "frame_id": f.frame_id,
        "type": "IDR",
        "bits": bits,
        "generation": s.generation,
        "refs": [r.as_dict() for r in s.refs],
    }


# --------------------------------------------------------------------------- #
# IDR 请求：原子完成参考代际切换
# --------------------------------------------------------------------------- #
def idr_request(state: Any) -> dict[str, Any]:
    s = _coerce_state(state)
    previous = s.generation
    # 先递增代际，让所有已排队的旧帧立刻失去引用资格。
    s.generation += 1
    cleared = clear_refs(s)
    invalidated = invalidate_queue(s)
    rate = reset_rate(s)
    reorder = reset_reorder(s)
    _log(
        s,
        "idr_request",
        previous_generation=previous,
        generation=s.generation,
    )
    return {
        "generation": s.generation,
        "previous_generation": previous,
        "refs": cleared["refs"],
        "cleared": cleared["cleared"],
        "dropped": invalidated["dropped"],
        "rate": rate,
        "reorder": reorder,
    }


# --------------------------------------------------------------------------- #
# B 帧重排序结构
# --------------------------------------------------------------------------- #
def bframe_order(state: Any, count: int = 9) -> dict[str, Any]:
    s = _coerce_state(state)
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        count = 9
    depth = max(0, s.reorder_depth)
    display = list(range(count))
    if depth == 0:
        encode_order = display
    else:
        step = depth + 1
        anchors = [i for i in display if i % step == 0]
        non_anchors = [i for i in display if i % step != 0]
        # 锚点先发，依赖锚点的 B 帧随后；每个 B 帧引用的锚点更早于它发出。
        encode_order = anchors + non_anchors
    return {
        "reorder_depth": depth,
        "display_order": display,
        "encode_order": encode_order,
    }


# --------------------------------------------------------------------------- #
# 校验 / 审计 / 指标 / 兜底
# --------------------------------------------------------------------------- #
def validation(state: Any) -> dict[str, Any]:
    s = _coerce_state(state)
    violations: list[dict[str, Any]] = []

    for ref in s.refs:
        if ref.generation != s.generation:
            violations.append(
                {"kind": "stale_ref", "frame_id": ref.frame_id, "generation": ref.generation}
            )

    valid = _valid_ref_ids(s)
    for frame in s.queue:
        if frame.status != PENDING:
            continue
        if frame.ref_generation != s.generation:
            violations.append(
                {
                    "kind": "stale_queue_frame",
                    "frame_id": frame.frame_id,
                    "ref_generation": frame.ref_generation,
                }
            )
        elif any(rid not in valid for rid in frame.ref_ids):
            violations.append({"kind": "missing_reference", "frame_id": frame.frame_id})

    if s.reorder_depth < s.base_reorder_depth:
        violations.append(
            {
                "kind": "reorder_depth_below_base",
                "depth": s.reorder_depth,
                "base": s.base_reorder_depth,
            }
        )

    return {
        "ok": not violations,
        "violations": violations,
        "generation": s.generation,
    }


def audit(state: Any) -> dict[str, Any]:
    s = _coerce_state(state)
    return {
        "generation": s.generation,
        "events": [dict(e) for e in s.events],
        "event_count": len(s.events),
        "digest": _digest(s),
    }


def fallback(state: Any, reason: Any = None, frame: Any = None) -> dict[str, Any]:
    s = _coerce_state(state)
    f = _coerce_frame(frame, s)
    # 参考失效时唯一安全的选择是强制下一帧为 IDR，并走新代际。
    _log(s, "fallback", reason=str(reason), frame_id=f.frame_id)
    return {
        "action": "force_idr",
        "reason": str(reason) if reason is not None else "invalid_reference",
        "frame_id": f.frame_id,
        "generation": s.generation,
        "reference": None,
    }


def metrics(state: Any) -> dict[str, Any]:
    s = _coerce_state(state)
    return _metrics(s)


def _metrics(state: EncoderState) -> dict[str, Any]:
    return {
        "generation": state.generation,
        "refs": len(state.refs),
        "queue_pending": sum(1 for f in state.queue if f.status == PENDING),
        "queue_total": len(state.queue),
        "reorder_depth": state.reorder_depth,
        "reorder_buffer": len(state.reorder_buffer),
        "complexity": state.complexity,
        "target_bits": state.target_bits,
        "bits_used": state.bits_used,
        **state.counters,
    }


def check(state: Any) -> dict[str, Any]:
    s = _coerce_state(state)
    report = validation(s)
    _log(s, "check", ok=report["ok"])
    return {"ok": report["ok"], "validation": report, "metrics": _metrics(s)}


# --------------------------------------------------------------------------- #
# 确定性：同一场景两次运行结果必须一致
# --------------------------------------------------------------------------- #
def determinism(state: Any = None) -> dict[str, Any]:
    first = _scenario_digest()
    second = _scenario_digest()
    return {
        "deterministic": first == second,
        "digest": first,
        "runs": 2,
    }


def _scenario_digest() -> str:
    s = EncoderState()
    idr_request(s)
    encode(s, {"type": "IDR"})
    for _ in range(3):
        latest = s.refs[-1].frame_id
        encode(s, {"type": "P", "ref_ids": [latest], "ref_generation": s.generation})
    retransmit_ref(s, {"ref_id": 1})
    check(s)
    return _digest(s)
