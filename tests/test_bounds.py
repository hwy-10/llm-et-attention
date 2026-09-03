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
    """★ T·S 를 인자로 받는다 — 고정값 하나로만 검증하지 않기 위해서."""
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
    """가이드 5.5절의 수치 예제를 그대로 재현한다.

    ⚠ 연쇄 비교(`a == b == c`)를 쓰지 않는다. 파이썬에서 그건
    `(a == b) and (b == c)` 라서, 기대값을 틀리게 고치면 **뒤쪽 비교가 먼저
    거짓**이 되어 실제 반환값이 무엇인지 끝내 알려주지 않는다.
    """
    q = np.array([50, -30, 20, -10])
    b = step_bounds(q)
    assert b.q_pos == 70
    assert b.q_neg == -40
    # 평면 2개 처리 후: R_2 = (2^6 - 1) * 70
    assert b.r(2) == 4410, f"R_2 = {b.r(2)} (기대 4410)"
    # 평면 4개 처리 후: R_4 = (2^4 - 1) * 70
    assert b.r(4) == 1050, f"R_4 = {b.r(4)} (기대 1050)"
    # 하한 계수도 같은 구조인지
    assert b.l_offset(2) == 63 * -40, f"L_2 offset = {b.l_offset(2)}"
    assert b.width(2) == 63 * (70 - (-40)), f"width(2) = {b.width(2)}"


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
    assert n_violations == 0, f"bounds violated in {n_violations}/{n_checked} cases"


def test_bracket_tightens_monotonically():
    """평면을 더 처리할수록 구간 폭이 좁아져야 한다.

    ★ 전제: Q+ − Q− > 0 이어야 한다.
      width(m) = W_m · (Q+ − Q−) 이므로 q 가 전부 0 이면 폭이 처음부터 0 이고
      '엄격히 감소'가 성립하지 않는다. 그 경우는
      test_zero_query_makes_the_interval_degenerate 가 따로 다룬다.
    """
    fq, p, _ = _setup(7)
    b = step_bounds(fq.stored[0])
    assert b.q_pos - b.q_neg > 0, "이 테스트의 전제 — 퇴화하지 않은 q"
    widths = [b.width(m) for m in range(9)]
    assert all(widths[i] > widths[i + 1] for i in range(8)), widths
    assert widths[-1] == 0, "uncertainty must be 0 after all planes are processed"


def test_bracket_exact_at_full_planes():
    """전 평면 처리 시 상한 == 하한 == 참값."""
    fq, p, exact = _setup(9)
    n_planes = p.shape[0]
    for s in range(p.shape[1]):
        s_full = accumulate(p[:, s, :])
        np.testing.assert_array_equal(s_full, exact[s])
        b = step_bounds(fq.stored[s], n_planes)
        assert b.r(n_planes) == 0 and b.l_offset(n_planes) == 0


# ---------------------------------------------------------------------------
# ★ 2026-08-28 반례 탐색에서 나온 것들 — 상한식이 성립하는 '조건'
# ---------------------------------------------------------------------------
def _bracket_violations(q, K, n_planes=8):
    """q, K(정수)로 전 평면에서 L_m <= s <= U_m 위반 수를 센다."""
    q = np.asarray(q, dtype=np.int64)
    K = np.asarray(K, dtype=np.int64)
    cum = cumulative_accumulate(partial_dots(q, to_bitplanes(K, n_planes))[:, 0, :])
    b = step_bounds(q, n_planes)
    s = K @ q
    return sum(
        int(np.sum((s < cum[m] + b.l_offset(m)) | (s > cum[m] + b.r(m))))
        for m in range(n_planes + 1)
    )


def test_bracket_survives_extreme_but_valid_inputs():
    """정의역 안에서는 깨지지 않는다 — 반례 탐색에서 확인된 것."""
    rng = np.random.default_rng(0)
    d = 8
    K = rng.integers(0, 256, (5, d))
    for q in (
        np.zeros(d, dtype=np.int64),          # 전부 0
        np.full(d, 127),                      # 전부 양수
        np.full(d, -127),                     # 전부 음수
        np.array([127] + [0] * (d - 1)),      # 한 원소만
        rng.integers(-127, 128, d),
    ):
        assert _bracket_violations(q, K) == 0
    for KK in (np.zeros((3, d), dtype=np.int64), np.full((3, d), 255)):
        assert _bracket_violations(rng.integers(-127, 128, d), KK) == 0


def test_out_of_range_key_is_rejected_before_it_breaks_the_bracket():
    """★ 상한식은 0 <= K < 2^n_planes 에서만 성립한다.

    to_bitplanes 가 uint16 으로 감싸며 조용히 자르면 비트평면이 K 를 대표하지
    못하고 그 순간 상한식이 깨진다 (K=256 -> 0, K=-1 -> 255).
    지금은 to_bitplanes 가 막아 주므로, 그 방어가 살아 있는지 확인한다.

    ⚠ verify_bracket() 은 이 경우를 못 잡는다. S_m 과 참값이 둘 다 오염되어
      서로는 일관되기 때문이다. 입구에서 막는 것 말고는 방법이 없다.
    """
    q = np.random.default_rng(1).integers(-127, 128, 8)
    for bad in (256, 300, 65535, -1):
        K = np.array([[bad] + [1] * 7], dtype=np.int64)
        try:
            _bracket_violations(q, K)
        except ValueError:
            continue
        raise AssertionError(f"K={bad} 가 조용히 통과했다 — 상한식이 깨진 채로 돈다")


def test_key_must_fit_the_plane_count():
    """평면 수보다 큰 값도 같은 이유로 막혀야 한다 (n=4 인데 K=255 등)."""
    q = np.random.default_rng(2).integers(-127, 128, 8)
    assert _bracket_violations(q, np.array([[15] + [1] * 7]), n_planes=4) == 0
    try:
        _bracket_violations(q, np.array([[255] + [1] * 7]), n_planes=4)
    except ValueError:
        return
    raise AssertionError("n_planes=4 에 K=255 가 통과했다")


def test_remaining_scale_rejects_out_of_range_m():
    """m=-1 이 511(평면 9개분)을 조용히 내던 자리."""
    from src.quantize import remaining_scale

    assert remaining_scale(0, 8) == 255 and remaining_scale(8, 8) == 0
    for bad in (-1, 9):
        try:
            remaining_scale(bad, 8)
        except ValueError:
            continue
        raise AssertionError(f"m={bad} 가 조용히 통과했다")


def test_zero_query_makes_the_interval_degenerate():
    """★ q 가 전부 0 이면 폭이 처음부터 0 이다.

    test_bracket_tightens_monotonically 는 엄격 부등식(>)이라 이 입력에서 깨진다.
    상한식 자체는 성립하므로(폭 0 인 구간이 참값을 품는다) 결함이 아니라
    '단조 감소'의 전제다.
    """
    b = step_bounds(np.zeros(8, dtype=np.int64))
    assert b.q_pos == 0 and b.q_neg == 0
    assert all(b.width(m) == 0 for m in range(9))
    assert _bracket_violations(np.zeros(8, dtype=np.int64),
                               np.random.default_rng(3).integers(0, 256, (4, 8))) == 0


def test_bracket_holds_for_varied_shapes():
    """★ _setup 의 고정값(T=128, S=4) 하나로만 검증하지 않는다.

    작은 T, 단일 스텝, 큰 T 를 함께 본다. 형상에 따라 브로드캐스팅이
    어긋나면 여기서 걸린다.
    """
    for T, S in ((4, 1), (17, 1), (33, 2), (256, 8)):
        fq, p, exact = _setup(11, T=T, S=S)
        n_planes, S_, T_ = p.shape
        assert (S_, T_) == (S, T)
        cum = cumulative_accumulate(p)
        for s in range(S):
            b = step_bounds(fq.stored[s], n_planes)
            for m in range(n_planes + 1):
                lo = cum[m, s] + b.l_offset(m)
                hi = cum[m, s] + b.r(m)
                assert np.all(exact[s] >= lo) and np.all(exact[s] <= hi), (T, S, m)
