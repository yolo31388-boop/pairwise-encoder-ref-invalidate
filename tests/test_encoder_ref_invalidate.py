import pytest
import media.encoder_ref_invalidate as mod

CASES = [
    ("clear_refs", (1,)),
    ("invalidate_queue", (1,)),
    ("reset_rate", (1,)),
    ("reset_reorder", (1,)),
    ("retransmit_ref", (1,)),
    ("ref_generation", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("encode", (1,)),
    ("metrics", (1,)),
    ("idr_request", (1,)),
    ("bframe_order", (1,)),
    ("check", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
