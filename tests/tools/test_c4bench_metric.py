import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("c4bench", ROOT / "tools/tensorfold-dsv41/dsv41_bench/c4bench.py")
c4bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c4bench)


def stream(tokens: int, first: float, last: float) -> dict:
    step = (last - first) / (tokens - 1)
    chunks = [(first + step * i, 4) for i in range(tokens)]
    return {"work": "w", "temp": 0.0, "first": first, "last": last, "tokens": tokens,
            "chars": 4 * tokens, "chunks": chunks, "tok_s": (tokens - 1) / (last - first)}


FAST = stream(40, 0.0, 4.0)
SLOW = stream(20, 1.0, 7.0)


def test_steady_covers_only_the_all_live_span():
    st = c4bench.steady([FAST, SLOW])
    assert st["all_live_s"] == 3.0
    assert abs(st["steady_tok_s"] - 13.0) < 0.01
    assert all(abs(a - b) < 0.01 for a, b in zip(st["per_stream_steady_tok_s"], [10.0, 3.0]))


def test_steady_window_excludes_the_endpoints_in_order():
    assert abs(c4bench.estimate_window_tokens(FAST, 1.0, 1.03) - 1.0) < 0.01
    assert c4bench.estimate_window_tokens(FAST, 4.0, 7.0) == 0.0


def test_the_tail_of_the_slowest_stream_pulls_the_aggregate_under_the_steady_rate():
    span = SLOW["last"] - FAST["first"]
    total = FAST["tokens"] + SLOW["tokens"]
    decode_aggregate = total / span
    st = c4bench.steady([FAST, SLOW])
    assert abs(decode_aggregate - 8.57) < 0.01
    assert st["steady_tok_s"] > decode_aggregate
    assert abs((SLOW["last"] - FAST["last"]) / span - 0.429) < 0.001


def test_cell_specs_match_the_m2bench_workload_shapes():
    order, temps, caps = c4bench.spec("mixed4")
    assert order == ["code", "prose", "tweet", "edit"]
    assert temps == [0.0, 0.7, 0.0, 0.7]
    assert caps == [384, 384, 384, 384]
    distinct, _, _ = c4bench.spec("code4d")
    assert distinct == ["code", "code2", "code3", "code4"]
    assert c4bench.spec("edit4")[2] == [384, 384, 384, 384]
