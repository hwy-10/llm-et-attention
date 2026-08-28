"""★ 네 설계의 불변식 검증 ★  배경지식 가이드 8.1절

    ①(baseline) = ②(seq) = ③(exact) 의 점수는 완전히 동일해야 한다.
    ③ 은 무손실 판정이므로 참 top-k 가 전부 살아남아야 한다.

이 테스트가 깨지면 이후 실험 결과는 전부 폐기해야 한다.
"""

import numpy as np

from src.accumulator import exact_int_scores, fold_and_quantize_query
from src.bounds import step_bounds
from src.designs import run_design
from src.masked_sum import partial_dots
from src.memory import BramSpec
from src.quantize import quantize_key, to_bitplanes
from src.schedule import ScheduleSpec
from src.threshold import topk_indices


def _case(seed=0, d=64, T=192):
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 1.0, size=(1, d))
    k = rng.normal(0, 1.0, size=(T, d))
    # 어텐션 sink 를 심어 종단이 실제로 일어나게 한다
    k[0] = 4.0 * (q[0] / np.linalg.norm(q[0])) * np.linalg.norm(k[0])
    k[5] = 3.0 * (q[0] / np.linalg.norm(q[0])) * np.linalg.norm(k[5])

    key = quantize_key(k, granularity="per_channel")
    fq = fold_and_quantize_query(q, key)
    p = partial_dots(fq.stored, to_bitplanes(key.stored, 8))[:, 0, :]
    exact = exact_int_scores(fq.stored, key)[0]
    return p, step_bounds(fq.stored[0]), exact


SCHED = ScheduleSpec(lanes=32, batch_size=32, two_phase_split=3, compaction_cost_cycles=2)
BRAM = BramSpec(word_tokens=32, decision_latency_planes=1)


def test_baseline_equals_true_dot_product():
    p, b, exact = _case(0)
    r = run_design("baseline", p, b, sched=SCHED, bram=BRAM)
    np.testing.assert_array_equal(r.scores_raw, exact)


def test_seq_equals_baseline_bit_exact():
    """★ 비트평면 순차 처리는 근사가 아니다. 완전히 같아야 한다 ★"""
    for seed in range(6):
        p, b, exact = _case(seed)
        a = run_design("baseline", p, b, sched=SCHED, bram=BRAM)
        s = run_design("seq", p, b, sched=SCHED, bram=BRAM)
        np.testing.assert_array_equal(a.scores_raw, s.scores_raw)
        np.testing.assert_array_equal(s.scores_raw, exact)


def test_exact_mode_preserves_topk():
    """★ 정확 모드 무손실 판정 ★ 참 top-k 가 전부 생존해야 한다."""
    for seed in range(8):
        for top_k in (1, 4, 8, 16):
            p, b, exact = _case(seed)
            r = run_design("exact", p, b, top_k=top_k, sched=SCHED, bram=BRAM)
            true_top = set(topk_indices(exact.astype(np.float64), top_k).tolist())
            survivors = set(np.flatnonzero(r.alive).tolist())
            assert true_top <= survivors, (
                f"seed={seed} k={top_k}: {true_top - survivors} of the true top-k were terminated"
            )


def test_exact_mode_survivor_scores_are_exact():
    """생존 토큰의 점수는 참값과 정확히 같아야 한다 (누산이 중단되지 않았으므로)."""
    p, b, exact = _case(2)
    r = run_design("exact", p, b, top_k=8, sched=SCHED, bram=BRAM)
    np.testing.assert_array_equal(r.scores_raw[r.alive], exact[r.alive])


def test_exact_mode_topk_ranking_matches():
    """마스킹된 점수의 top-k 순서가 참값 top-k 순서와 같아야 한다."""
    for seed in range(6):
        p, b, exact = _case(seed)
        r = run_design("exact", p, b, top_k=8, sched=SCHED, bram=BRAM)
        np.testing.assert_array_equal(
            topk_indices(exact.astype(np.float64), 8),
            topk_indices(r.scores, 8),
        )


def test_approx_with_zero_margin_equals_exact():
    """margin=0 인 근사 모드는 정확 모드와 완전히 같아야 한다."""
    for seed in range(5):
        p, b, _ = _case(seed)
        e = run_design("exact", p, b, top_k=8, margin=0.0, sched=SCHED, bram=BRAM)
        a = run_design("approx", p, b, top_k=8, margin=0.0, sched=SCHED, bram=BRAM)
        np.testing.assert_array_equal(e.alive, a.alive)
        np.testing.assert_array_equal(e.term_plane, a.term_plane)


def test_larger_margin_terminates_more():
    """여유값이 클수록 더 많이/일찍 종단되어야 한다 (가이드 5.6절)."""
    p, b, _ = _case(1)
    prev = None
    for margin in (0.0, 0.1, 0.3, 0.6):
        r = run_design("approx", p, b, top_k=8, margin=margin, sched=SCHED, bram=BRAM)
        cur = r.mean_term_plane
        if prev is not None:
            assert cur <= prev + 1e-9, f"termination got later at margin={margin}"
        prev = cur


def test_termination_never_drops_below_topk():
    """수치 동률로 전부 잘리는 사고가 없어야 한다."""
    for margin in (0.0, 0.5, 2.0):
        p, b, _ = _case(3)
        r = run_design("approx", p, b, top_k=8, margin=margin, sched=SCHED, bram=BRAM)
        assert int(r.alive.sum()) >= 8


def test_decision_latency_reduces_savings():
    """판정 지연이 커지면 읽기 절감이 줄어야 한다 (과대평가 방지 장치 확인)."""
    p, b, _ = _case(4)
    savings = []
    for lat in (0, 1, 2):
        r = run_design("exact", p, b, top_k=8, sched=SCHED,
                       bram=BramSpec(word_tokens=32, decision_latency_planes=lat))
        savings.append(r.reads.ideal_saving)
    assert savings[0] >= savings[1] >= savings[2], savings
