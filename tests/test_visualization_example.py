"""utils/visualization_example — 교보재가 거짓말을 하지 않는지 확인한다.

이 데모의 존재 이유는 "행렬 곱이 이렇게 계산된다"를 보여 주는 것이다.
따라서 검증해야 할 것은 성능이 아니라 **화면에 뜨는 분해가 실제 행렬 곱과
같은가** 하나뿐이다. 여기서 통과하지 못하면 교보재로서 값이 0 이다.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from utils.visualization_example import matmul, server


# ---------------------------------------------------------------------------
# 계산 코어
# ---------------------------------------------------------------------------

def test_multiply_matches_numpy():
    """수동 삼중 루프가 numpy 의 A @ B 와 같아야 한다 — 데모의 존재 근거."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        m, n, p = rng.integers(2, 9, size=3)
        A = rng.integers(-9, 10, size=(m, n))
        B = rng.integers(-9, 10, size=(n, p))
        got = matmul.multiply(A.tolist(), B.tolist())
        assert got == (A @ B).tolist()


def test_breakdown_total_is_the_cell():
    """항별 누산의 마지막 값이 곧 C[i][j] 여야 한다."""
    A = [[5, -3, 7, 2], [1, 4, -6, 8], [-2, 2, 3, -1]]
    B = [[2, 5, -1], [8, -4, 3], [6, 1, 7], [-5, 9, 2]]
    C = matmul.multiply(A, B)

    for i in range(len(A)):
        for j in range(len(B[0])):
            bd = matmul.breakdown(A, B, i, j)
            assert bd["total"] == C[i][j]
            assert bd["terms"][-1]["cumsum"] == C[i][j]


def test_breakdown_terms_are_the_row_and_column():
    """뽑아 놓은 벡터가 정말 A 의 i 행과 B 의 j 열이어야 한다.

    계산판이 보여 주는 두 줄이 엉뚱한 행·열이면 그림 전체가 거짓이 된다.
    """
    A = [[5, -3, 7, 2], [1, 4, -6, 8]]
    B = [[2, 5, -1], [8, -4, 3], [6, 1, 7], [-5, 9, 2]]
    bd = matmul.breakdown(A, B, 1, 2)

    assert bd["row"] == A[1]
    assert bd["col"] == [B[k][2] for k in range(len(B))]
    assert [t["a"] for t in bd["terms"]] == bd["row"]
    assert [t["b"] for t in bd["terms"]] == bd["col"]
    assert [t["prod"] for t in bd["terms"]] == [a * b for a, b in zip(bd["row"], bd["col"])]


def test_cumsum_is_a_running_total():
    """누적 줄이 진짜 부분합이어야 한다 (누산기 동작을 보이는 게 목적)."""
    A = [[3, -7, 1, 4]]
    B = [[2], [5], [-1], [8]]
    terms = matmul.breakdown(A, B, 0, 0)["terms"]

    acc = 0
    for t in terms:
        acc += t["prod"]
        assert t["cumsum"] == acc


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="cannot multiply"):
        matmul.multiply([[1, 2, 3]], [[1, 2], [3, 4]])


def test_ragged_and_non_integer_inputs_are_rejected():
    with pytest.raises(ValueError):
        matmul.multiply([[1, 2], [3]], [[1], [2]])
    with pytest.raises(ValueError):
        matmul.multiply([[1.5, 2]], [[1], [2]])
    with pytest.raises(TypeError):
        matmul.multiply([["a", 2]], [[1], [2]])


def test_single_row_query_is_allowed():
    """1 x d 쿼리 벡터가 거절되면 안 된다.

    MIN_DIM=2 는 스테퍼의 하한일 뿐 계산의 제약이 아니다. 디코드 스텝의
    q · K^T 가 정확히 이 모양이라, 여기서 막히면 정작 이 프로젝트가 다루는
    경우를 데모가 표현하지 못한다.
    """
    q = [[5, -3, 7, 2]]                        # 1 x 4
    K_T = [[200, 96], [32, 224], [176, 48], [16, 128]]   # 4 x 2

    C = matmul.multiply(q, K_T)
    assert C == (np.array(q) @ np.array(K_T)).tolist()
    assert len(C) == 1 and len(C[0]) == 2

    bd = matmul.breakdown(q, K_T, 0, 0)
    assert bd["row"] == q[0]
    assert bd["total"] == C[0][0]


def test_dimension_limits():
    with pytest.raises(ValueError):
        matmul.random_matrix(1, 3)
    with pytest.raises(ValueError):
        matmul.random_matrix(matmul.MAX_DIM + 1, 3)


def test_random_matrix_shape_seed_and_no_zeros():
    a = matmul.random_matrix(5, 4, seed=7)
    assert len(a) == 5 and all(len(r) == 4 for r in a)
    assert all(v != 0 for row in a for v in row)      # 0 이 섞이면 설명이 흐려진다
    assert a == matmul.random_matrix(5, 4, seed=7)    # 시드가 같으면 같은 값


def test_out_of_range_cell():
    A, B = [[1, 2], [3, 4]], [[1, 2], [3, 4]]
    with pytest.raises(IndexError):
        matmul.breakdown(A, B, 5, 0)
    with pytest.raises(IndexError):
        matmul.breakdown(A, B, 0, 5)


# ---------------------------------------------------------------------------
# 상태 조립
# ---------------------------------------------------------------------------

def test_build_state_shape_and_cell():
    A = matmul.random_matrix(5, 4, seed=1)
    B = matmul.random_matrix(4, 3, seed=2)
    st = server.build_state(A, B, 1, 2)

    assert st["dims"] == {"m": 5, "n": 4, "p": 3}
    assert len(st["C"]) == 5 and len(st["C"][0]) == 3
    assert st["cell"]["total"] == st["C"][1][2]


def test_build_state_without_selection():
    A = matmul.random_matrix(3, 3, seed=3)
    B = matmul.random_matrix(3, 3, seed=4)
    assert server.build_state(A, B)["cell"] is None


# ---------------------------------------------------------------------------
# HTTP — 실제로 서버를 띄워 본다
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_server():
    httpd = server.serve("127.0.0.1", 0, quiet=True)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=5)


def _post(base: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_page_and_assets_are_served(live_server):
    for path in ("/", "/static/style.css", "/static/app.js"):
        with urllib.request.urlopen(live_server + path, timeout=5) as resp:
            assert resp.status == 200
            assert resp.read()


def test_api_random_then_compute(live_server):
    st = _post(live_server, "/api/random", {"m": 5, "n": 4, "p": 3})
    assert st["dims"] == {"m": 5, "n": 4, "p": 3}

    out = _post(live_server, "/api/compute", {"A": st["A"], "B": st["B"], "i": 2, "j": 1})
    assert out["C"] == (np.array(st["A"]) @ np.array(st["B"])).tolist()
    assert out["cell"]["total"] == out["C"][2][1]


def test_api_rejects_bad_shapes(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(live_server, "/api/compute", {"A": [[1, 2, 3]], "B": [[1], [2]]})
    assert exc.value.code == 400


def test_static_path_traversal_is_blocked(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(live_server + "/static/../../../server.py", timeout=5)
    assert exc.value.code in (403, 404)


# ---------------------------------------------------------------------------
# 스케줄 데모 — src.schedule 을 그대로 구동하는지
# ---------------------------------------------------------------------------

from src import schedule as _src_schedule          # noqa: E402
from utils.visualization_example import schedule_demo  # noqa: E402


def test_read_live_is_monotone_and_plane0_is_full():
    """평면이 진행할수록 생존 토큰은 줄기만 한다. 평면 0 은 전부 읽는다.

    첫 평면에서 이미 죽는 토큰이 있으면 안 된다 — 아직 판정할 근거가 없다.
    """
    rl = schedule_demo.make_read_live(96, 8, survival=0.6, clustering=0.0)
    assert rl.shape == (8, 96)
    assert rl[0].all()

    counts = rl.sum(axis=1)
    assert all(counts[t] >= counts[t + 1] for t in range(len(counts) - 1))

    # 한번 죽은 토큰이 되살아나면 안 된다
    for i in range(rl.shape[1]):
        col = rl[:, i]
        assert list(col) == sorted(col, reverse=True)


def test_clustering_moves_words_but_not_logical_reads():
    """★ 이 데모의 주제. 배치만 바꾸면 논리적 읽기는 그대로, 워드는 달라진다."""
    kw = dict(n_tokens=128, n_planes=8, survival=0.55,
              lanes=32, batch_size=32, word_tokens=32)

    scattered = schedule_demo.run(dict(kw, clustering=0.0))
    clustered = schedule_demo.run(dict(kw, clustering=1.0))

    # 생존 토큰 수는 평면별로 완전히 같아야 한다
    assert ([p["live"] for p in scattered["per_plane"]]
            == [p["live"] for p in clustered["per_plane"]])
    assert scattered["reference"]["reads_ideal"] == clustered["reference"]["reads_ideal"]

    # 그런데 흩어져 있으면 batch 는 워드를 거의 못 줄인다
    assert (scattered["policies"]["batch"]["words_bram"]
            > clustered["policies"]["batch"]["words_bram"])

    # compaction 은 배치와 무관하게 같은 워드 수를 낸다 — "유일한 방법"인 이유
    assert (scattered["policies"]["compaction"]["words_bram"]
            == clustered["policies"]["compaction"]["words_bram"])


def test_run_calls_the_real_schedule_module(monkeypatch):
    """데모가 src.schedule.apply 를 실제로 부르는지 — 흉내 내면 의미가 없다."""
    calls = []
    real = _src_schedule.apply

    def spy(result, spec, bram, policy):
        calls.append(policy)
        return real(result, spec, bram, policy)

    monkeypatch.setattr(_src_schedule, "apply", spy)
    out = schedule_demo.run({"n_tokens": 32, "n_planes": 4})
    assert sorted(calls) == sorted(_src_schedule.POLICIES)
    assert set(out["policies"]) == set(_src_schedule.POLICIES)


def test_no_termination_matches_dense():
    """생존율 1.0 이면 아무도 종단되지 않으므로 compaction 의 워드는 dense 와 같다."""
    out = schedule_demo.run({"n_tokens": 96, "n_planes": 8, "survival": 1.0,
                             "lanes": 32, "word_tokens": 32})
    ref = out["reference"]
    assert ref["reads_ideal"] == ref["reads_dense"]
    assert out["policies"]["none"]["words_bram"] == ref["words_dense"]
    assert out["policies"]["compaction"]["words_bram"] == ref["words_dense"]


def test_out_of_range_params_are_clamped_not_crashed():
    """슬라이더 밖의 값이 와도 500 으로 죽지 않고 범위 안으로 잘린다."""
    out = schedule_demo.run({"n_tokens": 10**6, "n_planes": 0, "lanes": -5,
                             "survival": 9.9, "clustering": -3})
    sp = out["spec"]
    assert sp["n_tokens"] == schedule_demo.LIMITS["n_tokens"][1]
    assert sp["n_planes"] == schedule_demo.LIMITS["n_planes"][0]
    assert sp["lanes"] == schedule_demo.LIMITS["lanes"][0]
    assert 0.0 <= sp["survival"] <= 1.0
    assert 0.0 <= sp["clustering"] <= 1.0


def test_schedule_page_and_api(live_server):
    with urllib.request.urlopen(live_server + "/schedule", timeout=5) as resp:
        assert resp.status == 200
    for path in ("/static/schedule.css", "/static/schedule.js"):
        with urllib.request.urlopen(live_server + path, timeout=5) as resp:
            assert resp.status == 200

    out = _post(live_server, "/api/schedule",
                {"n_tokens": 96, "n_planes": 8, "survival": 0.62, "clustering": 0.0})
    assert set(out["policies"]) == set(_src_schedule.POLICIES)
    assert len(out["read_live"]) == 8 and len(out["read_live"][0]) == 96
    assert len(out["per_plane"]) == 8


def test_batch_never_exceeds_dense_when_nothing_terminates():
    """묶음이 없는 일을 만들어 내지 않는지 검증."""

    # 마지막 묶음은 토큰 4개다. 꽉 찬 묶음과 같은 값을 청구하면 dense 를 넘는다
    out = schedule_demo.run({"n_tokens": 100, "n_planes": 8, "survival": 1.0,
                             "lanes": 16, "batch_size": 32})

    assert out["policies"]["batch"]["cycles"] <= out["reference"]["dense_cycles"]

    # 종단이 하나도 없으면 미룰 것도 없으므로 정확히 같다
    assert out["policies"]["batch"]["cycles"] == out["reference"]["dense_cycles"]


# ---------------------------------------------------------------------------
# 실행 트레이스 — 그림이 apply() 와 같은 것을 말하는지
# ---------------------------------------------------------------------------

def test_trace_length_equals_apply_cycles():
    """★ 타임라인 길이 == apply() 의 사이클 수.

    이게 어긋나면 화면이 그리는 스케줄은 src/schedule.py 가 세는 스케줄이
    아니게 된다. 데모가 지켜야 할 가장 중요한 불변식이다.
    """
    from src.memory import BramSpec

    rl = schedule_demo.make_read_live(24, 6, survival=0.6, clustering=0.0)
    res = schedule_demo._as_step_result(rl)
    bram = BramSpec(word_tokens=4)

    for lanes, bs, split, cost in [(4, 8, 2, 2), (3, 7, 1, 0), (8, 24, 4, 3), (2, 5, 6, 1)]:
        spec = _src_schedule.ScheduleSpec(
            lanes=lanes, batch_size=bs, two_phase_split=split, compaction_cost_cycles=cost
        )
        for policy in _src_schedule.POLICIES:
            trace = schedule_demo.build_trace(rl, spec, policy)
            expect = _src_schedule.apply(res, spec, bram, policy).cycles
            assert len(trace) == expect, (policy, lanes, bs, split, cost)
            for cyc in trace:
                assert len(cyc["slots"]) == lanes
                assert len(cyc["waste"]) == lanes


def test_useful_slots_are_identical_across_policies():
    """유효 슬롯 수는 정책이 바꾸지 못한다 — 해야 할 일의 양은 종단이 정한다.

    화면이 "달라지는 건 몇 사이클에 담느냐뿐"이라고 말하는 근거다.
    """
    out = schedule_demo.run({"n_tokens": 16, "n_planes": 6, "lanes": 4,
                             "batch_size": 8, "two_phase_split": 2,
                             "compaction_cost_cycles": 2, "word_tokens": 4})
    useful = {n: out["traces"][n]["occupancy"]["slots_useful"] for n in out["policy_order"]}
    assert len(set(useful.values())) == 1, useful
    assert useful["none"] == out["reference"]["reads_ideal"]


def test_compaction_never_drags_terminated_tokens():
    """compaction 은 종단된 토큰을 끌고 가지 않는다 — 회색 슬롯이 0 이어야 한다."""
    out = schedule_demo.run({"n_tokens": 24, "n_planes": 6, "survival": 0.55,
                             "lanes": 4, "compaction_cost_cycles": 2, "word_tokens": 4})
    assert out["traces"]["compaction"]["occupancy"]["slots_waste"] == 0
    # 반대로 none 은 반드시 끌고 간다 (종단이 일어났다면)
    assert out["traces"]["none"]["occupancy"]["slots_waste"] > 0


def test_trace_slots_only_hold_tokens_of_that_batch():
    """batch 정책의 슬롯에는 그 배치의 토큰만 실려야 한다."""
    out = schedule_demo.run({"n_tokens": 20, "n_planes": 5, "lanes": 4, "batch_size": 8,
                             "word_tokens": 4})
    bs = out["spec"]["batch_size"]
    for cyc in out["traces"]["batch"]["cycles"]:
        if cyc["batch"] is None:
            continue
        lo, hi = cyc["batch"] * bs, (cyc["batch"] + 1) * bs
        for tok in cyc["slots"]:
            assert tok is None or lo <= tok < hi


def test_the_last_partial_batch_costs_only_its_own_width():
    """부분 묶음이 꽉 찬 묶음만큼 청구되지 않는지 검증.

    lanes=2, batch_size=8, n_tokens=20 이면 마지막 묶음은 토큰 4개다.
    """
    out = schedule_demo.run({"n_tokens": 20, "n_planes": 5, "survival": 1.0,
                             "lanes": 2, "batch_size": 8, "word_tokens": 4})

    # 토큰을 하나도 싣지 못하는 사이클이 없어야 한다
    empty = [c for c in out["traces"]["batch"]["cycles"] if all(t is None for t in c["slots"])]
    assert not empty, [c["batch"] for c in empty]

    # 꽉 찬 묶음 2개 x 4사이클 + 부분 묶음 1개 x 2사이클, 5평면 전부 생존
    assert out["policies"]["batch"]["cycles"] == 5 * (4 + 4 + 2)


def test_trace_is_omitted_when_too_large():
    """설정이 커지면 트레이스를 생략하되 사이클 수와 점유율은 여전히 준다.

    실제 스펙(512토큰 · 8평면 · 32레인)은 생략되지 않아야 하므로 방어선은
    그보다 훨씬 위에 있다. 여기서는 압축 비용을 최대로 올려 일부러 넘긴다.
    """
    out = schedule_demo.run({"n_tokens": 512, "n_planes": 16, "lanes": 128,
                             "survival": 0.6, "compaction_cost_cycles": 16,
                             "word_tokens": 32})
    tr = out["traces"]["compaction"]
    assert tr["omitted"] is True
    assert tr["cycles"] is None
    assert tr["n_cycles"] == out["policies"]["compaction"]["cycles"]
    assert tr["occupancy"]["slots_total"] > schedule_demo.MAX_TRACE_SLOTS


# ---------------------------------------------------------------------------
# 판정 — 어느 상황에서 어느 정책이 유리한가
# ---------------------------------------------------------------------------

_REAL = dict(n_tokens=128, n_planes=8, lanes=32, word_tokens=32,
             batch_size=32, two_phase_split=3, compaction_cost_cycles=2)


def test_limits_cover_the_real_project_spec():
    """슬라이더 범위가 config/hardware.yaml 의 실제 값을 담아야 한다.

    데모가 실제 스펙에 못 미치면 "우리 하드웨어에서는 어떻게 되나"에 답할 수 없다.
    """
    hw = schedule_demo.hardware_defaults()
    for key in ("lanes", "word_tokens", "batch_size", "two_phase_split",
                "compaction_cost_cycles"):
        lo, hi = schedule_demo.LIMITS[key]
        assert lo <= hw[key] <= hi, (key, hw[key], lo, hi)
    assert schedule_demo.LIMITS["n_tokens"][1] >= 512     # architecture.md 의 T
    assert schedule_demo.LIMITS["n_planes"][1] >= 8       # INT8


def test_hardware_defaults_match_the_config_file():
    """프리셋 '실제 스펙' 이 손으로 베낀 숫자가 아니라 설정 파일에서 나와야 한다."""
    from src.config import load_config

    cfg = load_config()
    hw = schedule_demo.hardware_defaults()
    assert hw["source"] == "config/hardware.yaml"
    assert hw["lanes"] == _src_schedule.spec_from_config(cfg).lanes
    assert hw["word_tokens"] == _src_schedule.bram_from_config(cfg).word_tokens
    assert hw["batch_size"] == _src_schedule.spec_from_config(cfg).batch_size


def test_real_spec_renders_a_timeline():
    """실제 스펙(문맥 512)에서도 타임라인을 생략하지 않아야 한다."""
    out = schedule_demo.run(dict(_REAL, n_tokens=512))
    for name in out["policy_order"]:
        assert out["traces"][name]["omitted"] is False


def test_verdict_clustered_prefers_batch():
    """뭉쳐서 죽으면 batch 가 compaction 만큼 얻으므로 batch 를 권한다."""
    v = schedule_demo.run(dict(_REAL, survival=0.6, clustering=1.0))["verdict"]
    assert v["situation"] == "clustered"
    assert v["recommended"] == "batch"
    assert v["best_words"] == "batch"


def test_verdict_scattered_prefers_compaction():
    """흩어져 죽으면 compaction 만 워드를 줄인다."""
    out = schedule_demo.run(dict(_REAL, survival=0.6, clustering=0.0))
    v = out["verdict"]
    assert v["situation"] == "scattered"
    assert v["recommended"] == "compaction"
    assert v["best_words"] == "compaction"
    # batch 는 이 상황에서 워드를 전혀 못 줄인다
    assert out["policies"]["batch"]["read_saving_bram"] < 0.01


def test_verdict_no_termination_avoids_compaction():
    """종단이 없으면 압축은 비용만 남긴다 — none/batch 를 권한다."""
    out = schedule_demo.run(dict(_REAL, survival=1.0, compaction_cost_cycles=6))
    v = out["verdict"]
    assert v["situation"] == "no_gain"
    assert v["recommended"] in ("none", "batch")
    # 그리고 실제로 compaction 이 dense 보다 느려야 한다
    assert out["policies"]["compaction"]["cycles"] > out["reference"]["dense_cycles"]


def test_verdict_reports_when_the_two_axes_disagree():
    """실제 스펙에서는 사이클 승자와 워드 승자가 다르다 — 그걸 숨기지 않는다."""
    v = schedule_demo.run(dict(_REAL, survival=0.62, clustering=0.0))["verdict"]
    assert v["axes_agree"] is False
    assert v["best_cycles"] != v["best_words"]


def test_verdict_situation_is_one_of_the_documented_four():
    """어떤 설정을 넣어도 판정은 화면의 상황별 표에 있는 넷 중 하나여야 한다."""
    seen = set()
    for surv in (0.3, 0.6, 0.9, 1.0):
        for clus in (0.0, 0.5, 1.0):
            for cost in (0, 2, 8):
                v = schedule_demo.run(dict(_REAL, survival=surv, clustering=clus,
                                           compaction_cost_cycles=cost))["verdict"]
                assert v["situation"] in schedule_demo._SITUATIONS
                assert v["recommended"] in _src_schedule.POLICIES
                seen.add(v["situation"])
    assert len(seen) >= 3, f"verdicts barely differ across presets: {seen}"


# ---------------------------------------------------------------------------
# BRAM 수행 시간 — 데모가 n_ports / mem_overlap 을 실제로 반영하는지
# ---------------------------------------------------------------------------

def test_demo_exposes_memory_cycles():
    """정책마다 메모리 사이클과 총 사이클이 나가야 한다 (화면의 시간축 막대가 쓴다)."""
    out = schedule_demo.run(dict(_REAL, n_ports=2, mem_overlap=True))
    for name in out["policy_order"]:
        p = out["policies"][name]
        assert p["memory_cycles"] == -(-p["words_bram"] // 2)      # ceil
        assert p["total_cycles_with_memory"] == max(p["cycles"], p["memory_cycles"])
        assert p["memory_bound"] == (p["memory_cycles"] > p["cycles"])


def test_demo_ports_and_overlap_reach_the_model():
    """슬라이더 값이 실제로 결과를 바꾸는지 — 예전 n_ports 처럼 죽은 값이 되면 안 된다."""
    a = schedule_demo.run(dict(_REAL, n_ports=1))
    b = schedule_demo.run(dict(_REAL, n_ports=8))
    assert a["policies"]["compaction"]["memory_cycles"] > b["policies"]["compaction"]["memory_cycles"]

    ov = schedule_demo.run(dict(_REAL, n_ports=1, mem_overlap=True))
    no = schedule_demo.run(dict(_REAL, n_ports=1, mem_overlap=False))
    for name in ov["policy_order"]:
        assert (no["policies"][name]["total_cycles_with_memory"]
                >= ov["policies"][name]["total_cycles_with_memory"])


def test_demo_memory_bound_preset_flips_the_recommendation():
    """★ 화면의 「메모리가 병목이 될 때」 프리셋이 실제로 판정을 뒤집는지.

    word_tokens 를 lanes 보다 좁게 하고 포트를 1개로 두면 메모리가 병목이 되고,
    그러면 사이클이 아니라 워드를 줄이는 정책을 골라야 한다.
    """
    normal = schedule_demo.run(dict(_REAL, word_tokens=32, n_ports=2))
    membound = schedule_demo.run(dict(_REAL, word_tokens=8, n_ports=1))

    assert not normal["verdict"]["all_memory_bound"]
    assert membound["verdict"]["all_memory_bound"]
    assert membound["verdict"]["situation"] == "memory_bound"

    # 연산 최소와 총 시간 최소가 서로 다른 정책이어야 의미가 있다
    assert membound["verdict"]["best_cycles"] != membound["verdict"]["best_total"]
    assert membound["verdict"]["recommended"] == membound["verdict"]["best_total"]


def test_demo_hardware_defaults_include_bram_settings():
    hw = schedule_demo.hardware_defaults()
    assert hw["n_ports"] >= 1
    assert isinstance(hw["mem_overlap"], bool)


# ---------------------------------------------------------------------------
# 함수 해부도 — 상자가 실제 코드에서 나오는지
# ---------------------------------------------------------------------------

from utils.visualization_example import anatomy  # noqa: E402


def test_module_index_lists_real_symbols():
    """목록이 src/schedule.py 의 실제 최상위 이름에서 나와야 한다."""
    idx = anatomy.module_index("schedule")
    names = {it["name"] for it in idx["items"]}

    for expected in ("apply", "dense_cycles", "baseline_cycles",
                     "spec_from_config", "bram_from_config", "ScheduleSpec"):
        assert expected in names, expected

    # 줄 번호 순으로 정렬되어야 화면에서 소스 순서대로 보인다
    lines = [it["lineno"] for it in idx["items"]]
    assert lines == sorted(lines)

    # 지금은 apply 하나만 펼쳐져 있다
    impl = {it["name"] for it in idx["items"] if it["implemented"]}
    assert impl == {"apply"}


def test_input_boxes_come_from_the_real_signature():
    """★ 입력 상자는 inspect.signature 에서 나온다 — 손으로 베낀 것이 아니다."""
    import inspect

    from src import schedule as _sched

    d = anatomy.describe("schedule", "apply")
    got = [i["name"] for i in d["inputs"]]
    want = list(inspect.signature(_sched.apply).parameters)
    assert got == want


def test_field_boxes_come_from_the_real_dataclasses():
    """필드 상자는 dataclasses.fields 에서 나온다."""
    import dataclasses

    from src import memory as _mem
    from src import schedule as _sched

    d = anatomy.describe()
    by_name = {i["name"]: i for i in d["inputs"]}

    assert [f["name"] for f in by_name["spec"]["fields"]] == [
        f.name for f in dataclasses.fields(_sched.ScheduleSpec)
    ]
    assert [f["name"] for f in by_name["bram"]["fields"]] == [
        f.name for f in dataclasses.fields(_mem.BramSpec)
    ]
    out = {o["name"]: o for o in d["outputs"]}
    assert [f["name"] for f in out["ScheduleResult"]["fields"]] == [
        f.name for f in dataclasses.fields(_sched.ScheduleResult)
    ]
    assert [f["name"] for f in out["ReadAccount"]["fields"]] == [
        f.name for f in dataclasses.fields(_mem.ReadAccount)
    ]


def test_branches_match_the_real_policy_tuple():
    """분기 상자는 schedule.POLICIES 그대로여야 한다."""
    d = anatomy.describe()
    assert [b["policy"] for b in d["branches"]] == list(_src_schedule.POLICIES)


def test_branch_code_is_cut_from_the_real_source():
    """★ 코드 인용이 복사본이 아니라 원본에서 잘라 온 것인지.

    복사본이면 코드가 바뀌어도 그림이 그대로 남는다. 실제 소스에 그 줄이
    들어 있는지 확인한다.
    """
    import inspect

    src = inspect.getsource(_src_schedule.apply)
    for b in anatomy.describe()["branches"]:
        assert b["code"].strip(), b["policy"]
        first = b["code"].splitlines()[0].strip()
        assert first in src, (b["policy"], first)
        # 잘라 온 블록의 모든 줄이 원본에 있어야 한다
        for line in b["code"].splitlines():
            if line.strip():
                assert line.strip() in src, (b["policy"], line)


def test_every_field_has_a_description():
    """★ 어긋남 감지 — 코드에 필드가 늘면 설명도 따라와야 한다.

    이 테스트가 깨지면 anatomy.py 의 _FIELD_DESC 에 설명을 추가하라는 뜻이다.
    화면에도 빨간 경고로 뜬다.
    """
    d = anatomy.describe()
    assert d["undocumented"] == [], (
        "fields with no description: " + ", ".join(d["undocumented"])
    )


def test_unimplemented_function_is_rejected_clearly():
    """아직 펼치지 않은 함수는 조용히 빈 화면이 아니라 명확한 오류를 낸다."""
    with pytest.raises(KeyError, match="not expanded"):
        anatomy.describe("schedule", "dense_cycles")


def test_anatomy_page_and_api(live_server):
    for path in ("/schedule_py", "/static/anatomy.css", "/static/anatomy.js"):
        with urllib.request.urlopen(live_server + path, timeout=5) as resp:
            assert resp.status == 200

    with urllib.request.urlopen(live_server + "/api/anatomy?module=schedule", timeout=5) as resp:
        idx = json.loads(resp.read().decode("utf-8"))
    assert idx["module"] == "src/schedule.py"

    with urllib.request.urlopen(
        live_server + "/api/anatomy?module=schedule&func=apply", timeout=5
    ) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    assert d["func"] == "apply"
    assert len(d["inputs"]) == 4
    assert len(d["branches"]) == 4

    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(
            live_server + "/api/anatomy?module=schedule&func=dense_cycles", timeout=5
        )
    assert exc.value.code == 404


# ---------------------------------------------------------------------------
# 용어 사전 — 참조가 깨지지 않았는지, 수치가 설정에서 나오는지
# ---------------------------------------------------------------------------

from utils.visualization_example import glossary  # noqa: E402


def test_glossary_ids_are_unique():
    g = glossary.build()
    ids = [t["id"] for t in g["terms"]]
    assert len(ids) == len(set(ids)), "duplicate ids: " + str(
        sorted({i for i in ids if ids.count(i) > 1})
    )


def test_glossary_cross_references_resolve():
    """★ 깨진 참조가 없어야 한다 — 화면에서 조용히 빈 버튼이 되기 때문이다."""
    g = glossary.build()
    ids = {t["id"] for t in g["terms"]}

    for t in g["terms"]:
        for c in t.get("confuse", []):
            assert c in ids, f"{t['id']}.confuse -> unknown term {c}"
    for p in g["pairs"]:
        assert p["a"] in ids, p
        assert p["b"] in ids, p


def test_every_term_has_a_valid_canvas_and_highlight():
    """캔버스와 강조 지정이 화면이 아는 형태여야 한다."""
    g = glossary.build()
    kinds = {
        "grid": {"column", "row", "chunk", "cell", "all", "live", "dead"},
        "bracket": {"marker"},
        "vector": {"q", "k", "pair", "bit"},
    }
    for t in g["terms"]:
        assert t["canvas"] in kinds, (t["id"], t["canvas"])
        hl = t["highlight"]
        key = "kind" if t["canvas"] != "vector" else "kind"
        assert hl[key] in kinds[t["canvas"]], (t["id"], hl)
        # 격자 강조는 축소판 범위 안에 있어야 한다
        if t["canvas"] == "grid":
            d = g["demo"]
            if hl["kind"] == "column":
                assert 0 <= hl["index"] < d["n_tokens"], t["id"]
            elif hl["kind"] == "row":
                assert 0 <= hl["index"] < d["n_planes"], t["id"]
            elif hl["kind"] == "cell":
                assert 0 <= hl["row"] < d["n_planes"] and 0 <= hl["col"] < d["n_tokens"], t["id"]
            elif hl["kind"] == "chunk":
                assert hl["unit"] in d, (t["id"], hl["unit"])
                assert hl["start"] + d[hl["unit"]] <= d["n_tokens"], t["id"]


def test_every_term_has_all_text_fields():
    g = glossary.build()
    for t in g["terms"]:
        for key in ("term", "en", "group", "one", "real", "detail"):
            assert t.get(key), (t["id"], key)
        assert t["group"] in g["groups"], t["id"]


def test_real_values_come_from_the_config_not_hardcoded():
    """★ 수치는 config/hardware.yaml 에서 읽어야 한다 — 손으로 베끼면 조용히 낡는다."""
    from src.config import load_config

    cfg = load_config()
    real = glossary.build()["real"]
    assert real["source"] == "config/hardware.yaml"
    assert real["lanes"] == _src_schedule.spec_from_config(cfg).lanes
    assert real["word_tokens"] == _src_schedule.bram_from_config(cfg).word_tokens
    assert real["n_ports"] == _src_schedule.bram_from_config(cfg).n_ports


def test_demo_grid_preserves_the_lane_word_relation():
    """★ 축소판이 요점을 왜곡하지 않아야 한다.

    실제 설정에서 lanes == word_tokens 이고, 그 '우연히 같음'이 이 페이지가
    보여 주려는 헷갈림의 원인이다. 축소판에서 둘을 다르게 잡으면 요점이 사라진다.
    """
    g = glossary.build()
    real, demo = g["real"], g["demo"]
    assert (demo["lanes"] == demo["word_tokens"]) == (real["lanes"] == real["word_tokens"])
    assert demo["lanes"] <= demo["n_tokens"]
    assert demo["word_tokens"] <= demo["n_tokens"]


def test_glossary_page_and_api(live_server):
    for path in ("/glossary", "/static/glossary.css", "/static/glossary.js"):
        with urllib.request.urlopen(live_server + path, timeout=5) as resp:
            assert resp.status == 200

    with urllib.request.urlopen(live_server + "/api/glossary", timeout=5) as resp:
        g = json.loads(resp.read().decode("utf-8"))
    assert len(g["terms"]) >= 15
    assert g["pairs"]
    # 세 캔버스가 모두 쓰여야 그림이 한쪽으로 쏠리지 않는다
    assert {t["canvas"] for t in g["terms"]} == {"grid", "bracket", "vector"}
# ---------------------------------------------------------------------------
# 코드 상관관계 — depgraph
# ---------------------------------------------------------------------------

def test_depgraph_reads_the_repository_not_a_hand_written_list():
    """★ 손으로 적은 그림은 코드가 바뀌면 조용히 낡는다.

    실제로 import 하는 것만 간선이 되는지, 저장소를 직접 뒤져 대조한다.
    """
    from utils.visualization_example import depgraph

    g = depgraph.build()
    ids = {n["id"] for n in g["nodes"]}
    pairs = {(e["source"], e["target"]) for e in g["edges"]}

    # 소스에 실제로 적혀 있는 import 는 간선으로 나와야 한다
    assert ("src/schedule.py", "src/memory.py") in pairs
    assert ("src/accumulator.py", "src/quantize.py") in pairs
    assert ("utils/metrics.py", "src/threshold.py") in pairs

    # 없는 관계를 지어내지 않는다
    assert ("src/quantize.py", "src/schedule.py") not in pairs
    assert ("src/memory.py", "src/designs.py") not in pairs

    # 담당 범위 13개가 전부 들어 있다
    for f in ("src/quantize.py", "src/masked_sum.py", "src/accumulator.py",
              "src/memory.py", "src/schedule.py", "utils/metrics.py",
              "utils/cost_model.py", "src/designs.py"):
        assert f in ids, f


def test_depgraph_marks_the_bridge():
    """두 팀이 주고받는 것은 read_live 하나뿐이다 — 그 간선만 따로 칠한다."""
    from utils.visualization_example import depgraph

    g = depgraph.build()
    bridges = {(e["source"], e["target"]) for e in g["edges"] if e["bridge"]}

    node = {n["id"]: n for n in g["nodes"]}

    # 브리지 = 우리 코드가 판정 결과를 받아 가는 지점. terminator 로 들어가는 간선뿐이다.
    assert bridges == {("src/schedule.py", "src/terminator.py"),
                       ("src/designs.py", "src/terminator.py")}, bridges
    assert node["src/terminator.py"]["scope"] == "bridge"
    assert node["src/memory.py"]["scope"] == "core"

    # ★ read_live 는 import 되는 이름이 아니라 StepResult 의 속성이다.
    # import 목록에서 찾던 조건이 한 번도 안 걸려 죽어 있었다 — 원문에서 찾는다.
    assert all(g["summary"]["bridge_name"] not in e["names"] for e in g["edges"])
    assert set(g["summary"]["touch_bridge"]) == {"decode_loop", "memory", "schedule", "terminator"}


def test_depgraph_layout_has_no_overlap():
    """좌표는 파이썬이 준다 — 겹치면 화면에서 상자가 포개진다."""
    from utils.visualization_example import depgraph

    g = depgraph.build()
    seen = set()
    for n in g["nodes"]:
        key = (n["x"], n["y"])
        assert key not in seen, f"{n['id']} 자리가 겹친다"
        seen.add(key)
        assert 0 <= n["x"] <= g["layout"]["width"]
        assert 0 <= n["y"] <= g["layout"]["height"]

    # 의존 대상은 항상 왼쪽에 있어야 화살표 방향이 뜻을 갖는다
    node = {n["id"]: n for n in g["nodes"]}
    for e in g["edges"]:
        assert node[e["target"]]["col"] < node[e["source"]]["col"], (
            f"{e['source']} -> {e['target']}: 의존 대상이 오른쪽에 있다"
        )


def test_depgraph_counts_lines_from_the_files():
    """줄 수를 손으로 적어 두면 파일이 자라도 그림은 그대로다."""
    from src.config import PROJECT_ROOT
    from utils.visualization_example import depgraph

    for n in depgraph.build()["nodes"]:
        text = (PROJECT_ROOT / n["id"]).read_text(encoding="utf-8")
        real = text.count(chr(10)) + 1
        assert n["lines"] == real, n["id"]


def test_depgraph_flags_files_no_test_imports():
    """검사가 덮지 않는 담당 파일을 숨기지 않고 세어 낸다."""
    from utils.visualization_example import depgraph

    g = depgraph.build()
    node = {n["id"]: n for n in g["nodes"]}

    assert "test_memory" in node["src/memory.py"]["tests"]
    assert "test_schedule" in node["src/schedule.py"]["tests"]
    assert isinstance(g["summary"]["untested"], list)


def test_depgraph_page_and_api_are_served(live_server):
    """페이지와 API 가 실제로 나가는지."""
    for path in ("/depgraph", "/static/depgraph.css", "/static/depgraph.js"):
        with urllib.request.urlopen(live_server + path, timeout=5) as resp:
            assert resp.status == 200
            assert resp.read()

    with urllib.request.urlopen(live_server + "/api/depgraph", timeout=10) as resp:
        payload = json.loads(resp.read())
    assert payload["nodes"] and payload["edges"]
    assert set(payload["scopes"]) >= {"core", "extra", "integration", "bridge"}


def test_every_page_links_to_every_other_page():
    """길잡이가 끊기면 새 페이지를 아무도 못 찾는다."""
    from utils.visualization_example import server as srv

    pages = {
        "index.html": "/", "schedule.html": "/schedule",
        "schedule_py.html": "/schedule_py", "glossary.html": "/glossary",
        "depgraph.html": "/depgraph",
    }
    for name in pages:
        text = (srv.STATIC_DIR / name).read_text(encoding="utf-8")
        for other, href in pages.items():
            if other == name:
                assert 'aria-current="page"' in text, name
            else:
                assert f'href="{href}"' in text, f"{name} 에 {href} 링크가 없다"
