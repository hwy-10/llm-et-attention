"""★ 조기 종단 무손실성의 수학적 근거 검증 ★

배경지식 가이드 5.4절:

    L_m  ≤  s_j  ≤  S_m + R_m      (모든 m, 모든 j 에 대해)

이 부등식이 깨지면 종단이 무손실이 아니게 되고, 설계 전체가 무너진다.
랜덤 케이스를 대량으로 던져 확인한다.
"""

import numpy as np

from src.accumulator import accumulate, cumulative_accumulate, exact_int_scores
from src.bounds import batch_step_bounds, step_bounds
from src.masked_sum import partial_dots
from src.quantize import quantize_key, to_bitplanes
from src.accumulator import fold_and_quantize_query


def _setup(seed, d=64, T=128, S=4):
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 1.0, size=(S, d))
    k = rng.normal(0, 1.0, size=(T, d))
    key = quantize_key(k, granularity="per_channel")
    fq = fold_and_quantize_query(q, key)
    planes = to_bitplanes(key.stored, 8)
    p = partial_dots(fq.stored, planes)             # (8, S, T)
    exact = exact_int_scores(fq.stored, key)        # (S, T)
    return fq, p, exact


def test_qpos_qneg_definition():
    q = np.array([50, -30, 20, -10, 0])
    b = step_bounds(q)
    assert b.q_pos == 70
    assert b.q_neg == -40


def test_guide_numeric_example():
    """가이드 5.5절의 수치 예제를 그대로 재현한다."""
    q = np.array([50, -30, 20, -10])
    b = step_bounds(q)
    assert b.q_pos == 70
    # 평면 2개 처리 후: R_2 = (2^6 - 1) * 70 = 4410
    assert b.r(2) == 63 * 70 == 4410
    # 평면 4개 처리 후: R_4 = (2^4 - 1) * 70 = 1050
    assert b.r(4) == 15 * 70 == 1050


def test_bracket_holds_everywhere():
    """★ 핵심 테스트 ★ 랜덤 케이스에서 L_m ≤ s ≤ U_m 이 항상 성립."""
    n_violations = 0
    n_checked = 0
    for seed in range(12):
        fq, p, exact = _setup(seed)
        n_planes, S, T = p.shape
        cum = cumulative_accumulate(p)              # (9, S, T)
        q_pos, q_neg = batch_step_bounds(fq.stored, n_planes)
        for s in range(S):
            b = step_bounds(fq.stored[s], n_planes)
            assert b.q_pos == q_pos[s] and b.q_neg == q_neg[s]
            for m in range(n_planes + 1):
                lo = cum[m, s] + b.l_offset(m)
                hi = cum[m, s] + b.r(m)
                n_checked += T
                n_violations += int(np.sum((exact[s] < lo) | (exact[s] > hi)))
    assert n_violations == 0, f"{n_violations}/{n_checked} 케이스에서 상하한이 깨졌다"


def test_bracket_tightens_monotonically():
    """평면을 더 처리할수록 구간 폭이 좁아져야 한다."""
    fq, p, _ = _setup(7)
    b = step_bounds(fq.stored[0])
    widths = [b.width(m) for m in range(9)]
    assert all(widths[i] > widths[i + 1] for i in range(8)), widths
    assert widths[-1] == 0, "전 평면 처리 후 불확실성은 0 이어야 한다"


def test_bracket_exact_at_full_planes():
    """전 평면 처리 시 상한 == 하한 == 참값."""
    fq, p, exact = _setup(9)
    n_planes = p.shape[0]
    for s in range(p.shape[1]):
        s_full = accumulate(p[:, s, :])
        np.testing.assert_array_equal(s_full, exact[s])
        b = step_bounds(fq.stored[s], n_planes)
        assert b.r(n_planes) == 0 and b.l_offset(n_planes) == 0
