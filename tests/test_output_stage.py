"""출력 경로 검증 — ARCHITECTURE.md 7.1.5 의 항목 4개를 그대로 옮긴 것.

  [1] score_last 가 정확히 마지막 유효 토큰에서만 1 인가
  [2] 생존 수가 OUT_BUF 를 넘는 스텝에서 CNT_TRUNC 가 증가하는가
  [3] 잘라낸 뒤에도 참 top-K_TOP 이 남는가
  [4] zero-point 보정 전후 순위가 같은가
"""

import warnings

import numpy as np
import pytest

from src.bounds import StepBounds
from src.config import load_config
from src.output_stage import (
    OutputCounters,
    OutputSpec,
    emit_step,
    output_spec_from_config,
)
from src.quantize import N_PLANES
from src.terminator import StepResult, run_step
from src.threshold import ThetaPolicy


def _result(alive, s_int):
    """StepResult 를 최소 필드만 채워 만든다. 출력 단계는 alive/s_int 만 본다."""
    alive = np.asarray(alive, dtype=bool)
    n = alive.size
    return StepResult(
        s_int=np.asarray(s_int, dtype=np.int64),
        alive=alive,
        term_plane=np.where(alive, N_PLANES, 4).astype(np.int32),
        read_live=np.ones((N_PLANES, n), dtype=bool),
        theta_trace=np.zeros(N_PLANES),
        live_count=np.full(N_PLANES, n),
        n_active=n,
    )


def _real_step(rng, n_tokens=200, top_k=16, margin=0.0):
    """실제 종단 경로를 한 번 돌려 생존 집합을 얻는다."""
    p = rng.integers(0, 400, size=(N_PLANES, n_tokens))
    q = rng.integers(-64, 64, size=64)
    b = StepBounds(q_pos=int(q[q > 0].sum()), q_neg=int(q[q < 0].sum()))
    pol = ThetaPolicy(top_k=top_k, margin=margin, margin_mode="relative_width")
    return run_step(p, b, pol, decision_latency=1)


# --- [1] score_last ---------------------------------------------------------

def test_score_last_is_set_only_on_final_token():
    st = emit_step(_result([1, 0, 1, 1, 0], [10, 99, 20, 30, 99]), OutputSpec(out_buf=8))
    assert st.n_emit == 3
    assert st.last.tolist() == [False, False, True]
    assert st.last.sum() == 1


def test_score_last_is_all_zero_when_nothing_survives():
    # 다운스트림은 score_valid 가 한 번도 안 서는 것으로 빈 스텝을 안다.
    st = emit_step(_result([0, 0, 0], [1, 2, 3]), OutputSpec(out_buf=8))
    assert st.n_emit == 0
    assert st.last.tolist() == []


def test_emitted_indices_are_ascending_and_are_the_survivors():
    # 하드웨어는 토큰 번호 순으로 훑으며 내보낸다. 정렬해서 내보내지 않는다.
    st = emit_step(_result([0, 1, 1, 0, 1], [5, 70, 10, 5, 40]), OutputSpec(out_buf=8))
    assert st.idx.tolist() == [1, 2, 4]
    assert np.all(np.diff(st.idx) > 0)


def test_last_tracks_emitted_count_not_alive_count_when_truncated():
    # 자른 뒤의 마지막에 서야 한다. 자르기 전 생존 수를 쓰면 스트림이 안 끝난다.
    st = emit_step(_result([1] * 10, list(range(10))), OutputSpec(out_buf=4))
    assert st.n_alive == 10 and st.n_emit == 4
    assert st.last.sum() == 1 and st.last[-1]


# --- [2] CNT_TRUNC ----------------------------------------------------------

def test_cnt_trunc_increments_only_when_buffer_overflows():
    c = OutputCounters()
    emit_step(_result([1] * 4, [1, 2, 3, 4]), OutputSpec(out_buf=8), counters=c)
    assert c.cnt_trunc == 0 and c.tokens_dropped == 0

    emit_step(_result([1] * 12, list(range(12))), OutputSpec(out_buf=8), counters=c)
    assert c.cnt_trunc == 1 and c.tokens_dropped == 4
    assert c.max_alive == 12


def test_cnt_trunc_counts_steps_and_tokens_separately():
    # 스텝 수와 토큰 수는 다른 값이다. 논문에는 둘 다 필요하다.
    c = OutputCounters()
    for _ in range(3):
        emit_step(_result([1] * 10, list(range(10))), OutputSpec(out_buf=8), counters=c)
    assert c.cnt_trunc == 3          # 스텝 3번
    assert c.tokens_dropped == 6     # 매번 2개씩


def test_counters_record_alive_histogram_for_buffer_sizing():
    # OUT_BUF 를 정한 근거가 이 히스토그램이다. 재현할 수 있어야 한다.
    c = OutputCounters()
    for n in (16, 20, 20, 31):
        emit_step(_result([1] * n, list(range(n))), OutputSpec(out_buf=32), counters=c)
    assert c.hist == {16: 1, 20: 2, 31: 1}
    assert c.max_alive == 31 and c.cnt_trunc == 0


# --- [3] 자르기가 참 top-k 를 보존한다 --------------------------------------

def test_truncation_keeps_the_highest_scores():
    st = emit_step(_result([1] * 6, [10, 50, 20, 60, 30, 40]), OutputSpec(out_buf=3))
    assert st.idx.tolist() == [1, 3, 5]           # 점수 60, 50, 40
    assert sorted(st.data.tolist()) == [40, 50, 60]


def test_truncation_preserves_true_topk():
    """7.1.5-[3]. 실측이 아니라 성질로 확인한다.

    출력 시점에는 잔여 계수 (2^(n-m)-1) 가 m=n 이라 0 이므로 상한 = 확정 점수다.
    따라서 '상한 상위로 자른다' = '점수 상위로 자른다' 이고,
    out_buf >= top_k 인 한 참 top-k 는 절대 잘리지 않는다.
    """
    rng = np.random.default_rng(20260829)
    top_k = 16
    for _ in range(30):
        res = _real_step(rng, top_k=top_k)
        true_topk = set(np.argsort(-res.s_int, kind="stable")[:top_k].tolist())
        # 종단 자체가 무손실이어야 자르기를 논할 수 있다 (전제 확인)
        assert true_topk <= set(np.flatnonzero(res.alive).tolist())

        st = emit_step(res, OutputSpec(out_buf=top_k))     # 최악: 하한까지 조인다
        assert true_topk == set(st.idx.tolist())


def test_ties_are_broken_by_index_so_hardware_is_deterministic():
    # 동점에서 순서가 흔들리면 RTL 과 골든모델이 어긋난다.
    st = emit_step(_result([1] * 4, [7, 7, 7, 7]), OutputSpec(out_buf=2))
    assert st.idx.tolist() == [0, 1]


def test_no_truncation_when_buffer_is_large_enough():
    rng = np.random.default_rng(7)
    c = OutputCounters()
    for _ in range(40):
        emit_step(_real_step(rng, top_k=16), OutputSpec(out_buf=32), counters=c)
    # 7.1.4 의 "2 x K_TOP 을 한 번도 넘지 않았다" 를 회귀로 고정한다.
    assert c.cnt_trunc == 0, f"넘친 스텝 {c.cnt_trunc}회, 최대 생존 {c.max_alive}"


def test_buffer_smaller_than_topk_is_a_design_error():
    # 가드가 최소 top_k 개를 남기므로 out_buf < top_k 면 매 스텝 자른다.
    assert not OutputSpec(out_buf=8).capacity_ok(16)
    assert OutputSpec(out_buf=32).capacity_ok(16)


# --- [4] zero-point 보정 ----------------------------------------------------

def test_zero_point_is_rank_invariant():
    """모든 토큰에 같은 상수라 순위가 안 변한다. 그래서 term_ctrl 은 정수만 본다."""
    res = _result([1] * 5, [100, 300, 200, 500, 400])
    a = emit_step(res, OutputSpec(out_buf=8), zp_correction=0)
    b = emit_step(res, OutputSpec(out_buf=8), zp_correction=1234)
    assert np.array_equal(np.argsort(-a.data), np.argsort(-b.data))
    assert np.array_equal(a.data - 1234, b.data)


def test_zero_point_does_not_change_which_tokens_are_truncated():
    res = _result([1] * 6, [10, 50, 20, 60, 30, 40])
    a = emit_step(res, OutputSpec(out_buf=3), zp_correction=0)
    b = emit_step(res, OutputSpec(out_buf=3), zp_correction=-9999)
    assert a.idx.tolist() == b.idx.tolist()


def test_zero_point_correction_matches_accumulator():
    """골든모델의 복원식과 같은 값을 쓴다 — 보정을 두 번 빼거나 안 빼면 걸린다."""
    from src.accumulator import fold_and_quantize_query
    from src.quantize import quantize_key

    rng = np.random.default_rng(3)
    key = quantize_key(rng.normal(size=(12, 64)))
    fq = fold_and_quantize_query(rng.normal(size=(1, 64)), key)
    s_int = (fq.stored.astype(np.int64) @ key.stored.astype(np.int64).T)[0]

    st = emit_step(_result([1] * 12, s_int), OutputSpec(out_buf=16),
                   zp_correction=int(fq.zp_correction[0]))
    # to_real_scores 가 하는 일 = (s_int - zp) * scale.  스트림은 그 괄호 안이다.
    assert np.array_equal(st.data, s_int - int(fq.zp_correction[0]))
    assert int(fq.zp_correction[0]) != 0     # 보정이 0 이면 이 시험이 무의미하다


# --- 비트폭 / 배선 ----------------------------------------------------------

def test_index_wider_than_idx_bits_is_rejected():
    # N_MAX 를 2048 로 늘리면서 idx_bits 를 9 로 두면 여기서 걸린다.
    big = _result([0] * 512 + [1], list(range(513)))
    with pytest.raises(ValueError, match="score_idx"):
        emit_step(big, OutputSpec(out_buf=8, idx_bits=9))
    emit_step(big, OutputSpec(out_buf=8, idx_bits=11))       # 11b 면 통과


def test_score_wider_than_data_bits_is_rejected():
    with pytest.raises(ValueError, match="score_data"):
        emit_step(_result([1], [1 << 23]), OutputSpec(out_buf=8, data_bits=24))
    emit_step(_result([1], [(1 << 23) - 1]), OutputSpec(out_buf=8, data_bits=24))


def test_zero_point_can_push_a_score_over_the_width():
    # 보정 전에는 들어가는데 보정 후에 넘칠 수 있다. 검사는 보정 뒤에 해야 한다.
    with pytest.raises(ValueError, match="score_data"):
        emit_step(_result([1], [0]), OutputSpec(out_buf=8, data_bits=24),
                  zp_correction=1 << 23)


def test_config_wiring_is_connected():
    """OutputSpec 기본값은 config 와 일부러 다르다. 같아지면 배선이 끊겨도 모른다.

    ★ N_MAX 를 바꾸면 out_buf / idx_bits 도 따라와야 한다. 값을 하드코딩하지 않고
      config 의 seq_len 에서 유도해 비교한다 — 셋이 어긋나면 여기서 걸린다.
    """
    cfg = load_config()
    n_max = cfg.seq_len
    spec = output_spec_from_config(cfg)
    assert spec.out_buf == n_max, f"out_buf {spec.out_buf} != N_MAX {n_max}"
    assert spec.out_buf != OutputSpec().out_buf, "기본값과 같아지면 배선 단절을 못 잡는다"
    assert (1 << spec.idx_bits) >= n_max, (
        f"idx_bits {spec.idx_bits} 로는 {1 << spec.idx_bits} 까지만 — N_MAX {n_max} 를 못 담는다"
    )
    assert spec.data_bits == 24


def test_adopted_config_cannot_truncate():
    """★ 채택 설정은 N_MAX 를 담으므로 자르기가 원리적으로 불가능하다.

    2 x K_TOP = 32 로 되돌리면 여기서 걸린다 — 실측 최대가 34 다.
    """
    cfg = load_config()
    spec = output_spec_from_config(cfg)
    assert spec.is_lossless(n_max=cfg.seq_len)
    assert not OutputSpec(out_buf=2 * 16).is_lossless(n_max=cfg.seq_len)


def test_two_times_topk_is_not_enough_in_the_worst_seed():
    """8/28 사양(OUT_BUF = 2 x K_TOP)이 왜 틀렸는지를 반례로 고정한다.

    "실측 최대 31, 초과 0회" 는 시드 하나만 본 값이었다.
    """
    rng = np.random.default_rng(0)
    c = OutputCounters()
    for _ in range(120):
        emit_step(_real_step(rng, n_tokens=512, top_k=16), OutputSpec(out_buf=4096),
                  counters=c)
    assert c.max_alive > 2 * 16, f"최대 생존 {c.max_alive} — 32 를 넘는 경우를 못 찾았다"

    # 같은 스텝들을 OUT_BUF=32 로 다시 흘리면 실제로 잘린다
    rng = np.random.default_rng(0)
    c32 = OutputCounters()
    for _ in range(120):
        emit_step(_real_step(rng, n_tokens=512, top_k=16), OutputSpec(out_buf=32),
                  counters=c32)
    assert c32.cnt_trunc > 0 and c32.tokens_dropped > 0


def test_missing_config_section_warns_instead_of_silently_defaulting():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        spec = output_spec_from_config({"hardware": {}})
    assert spec.out_buf == OutputSpec().out_buf
    assert any("output" in str(x.message) for x in w)


def test_out_buf_zero_is_rejected():
    with pytest.raises(ValueError, match="out_buf"):
        emit_step(_result([1], [1]), OutputSpec(out_buf=0))


def test_alive_must_be_one_dimensional():
    res = _result([1, 1], [1, 2])
    res.alive = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="1-D"):
        emit_step(res, OutputSpec(out_buf=8))
