"""시프트 누산기 검증.

검증 대상
1. 평면별 가중치 누산
2. 단계별 누산값
3. 비트평면 방식과 일반 INT8 내적의 일치
4. 64차원 무작위 입력
"""

import numpy as np
import pytest

from src.accumulator import (
    accumulate,
    cumulative_accumulate,
    exact_int_scores,
    fold_and_quantize_query,
    to_real_scores,
)
from src.masked_sum import partial_dots
from src.quantize import KeyQuant, plane_weights, quantize_key, to_bitplanes


def test_accumulate_manual_four_planes():
    """4비트 수동 예제로 각 단계의 누산값을 검증한다."""

    # 평면별 부분 내적값
    # MSB부터 순서대로 P3=3, P2=-2, P1=5, P0=1
    partials = np.array([
        [[3]],
        [[-2]],
        [[5]],
        [[1]],
    ], dtype=np.int32)

    # 4비트 가중치는 [8, 4, 2, 1]
    expected_by_m = [
        0,   # m=0
        24,  # 8×3
        16,  # 8×3 + 4×(-2)
        26,  # 8×3 + 4×(-2) + 2×5
        27,  # 8×3 + 4×(-2) + 2×5 + 1×1
    ]

    for m, expected in enumerate(expected_by_m):
        got = accumulate(partials, m=m)

        np.testing.assert_array_equal(
            got,
            np.array([[expected]], dtype=np.int64),
        )

    # m=None이면 모든 평면을 누산한다.
    np.testing.assert_array_equal(
        accumulate(partials),
        np.array([[27]], dtype=np.int64),
    )


def test_cumulative_accumulate_manual():
    """모든 중간 누산값이 수동 계산과 일치하는지 검증한다."""

    partials = np.array([
        [[3]],
        [[-2]],
        [[5]],
        [[1]],
    ], dtype=np.int32)

    got = cumulative_accumulate(partials)

    expected = np.array([
        [[0]],
        [[24]],
        [[16]],
        [[26]],
        [[27]],
    ], dtype=np.int64)

    assert got.shape == (5, 1, 1)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, expected)


def test_bitplane_partial_accumulate_manual():
    """비트평면 변환→부분 내적→누산 결과를 수동 정답과 비교한다."""

    # Query 1개, 차원 4
    q_stored = np.array([
        [2, -3, 5, 1],
    ], dtype=np.int16)

    # Key 토큰 2개, 각 Key 차원 4개
    K_stored = np.array([
        [170, 85, 240, 15],
        [1, 2, 3, 4],
    ], dtype=np.uint8)

    # 1. Key를 8개 비트평면으로 변환
    planes = to_bitplanes(K_stored)

    # 2. 각 평면의 부분 내적 계산
    partials = partial_dots(q_stored, planes)

    # 3. [128, 64, ..., 1] 가중치로 전체 누산
    got = accumulate(partials)

    # 토큰 0:
    # 2×170 + (-3)×85 + 5×240 + 1×15 = 1300
    #
    # 토큰 1:
    # 2×1 + (-3)×2 + 5×3 + 1×4 = 15
    expected = np.array([
        [1300, 15],
    ], dtype=np.int64)

    assert planes.shape == (8, 2, 4)
    assert partials.shape == (8, 1, 2)
    assert got.shape == (1, 2)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, expected)


def test_accumulated_score_equals_exact_int_score():
    """비트평면 누산값과 일반 INT8 행렬곱 결과를 비교한다."""

    # Query 2개, 차원 4
    q_stored = np.array([
        [2, -3, 5, 1],
        [-1, 4, 2, -2],
    ], dtype=np.int16)

    # Key 토큰 3개, 각 Key 차원 4개
    K_stored = np.array([
        [170, 85, 240, 15],
        [1, 2, 3, 4],
        [255, 0, 128, 64],
    ], dtype=np.uint8)

    key = KeyQuant(
        stored=K_stored,
        scale=np.array(1.0),
        zero_point=np.array(0),
    )

    # 비트평면 방식
    planes = to_bitplanes(K_stored)
    partials = partial_dots(q_stored, planes)

    score_bitplane = accumulate(partials)

    # 일반 INT8 행렬곱으로 기준 정답 계산
    score_exact = exact_int_scores(q_stored, key)

    # 두 방식의 결과가 일치하는지 검증
    np.testing.assert_array_equal(
        score_bitplane,
        score_exact,
    )


def test_random_64d_accumulation():
    """실제 설정과 같은 head_dim D=64 무작위 입력으로 검증한다."""

    rng = np.random.default_rng(20)

    # Query 3개, 차원 64
    q_stored = rng.integers(
        -127,
        128,
        size=(3, 64),
        dtype=np.int16,
    )

    # Key 토큰 37개, 차원 64
    K_stored = rng.integers(
        0,
        256,
        size=(37, 64),
        dtype=np.uint16,
    ).astype(np.uint8)

    planes = to_bitplanes(K_stored)
    partials = partial_dots(q_stored, planes)

    score_bitplane = accumulate(partials)

    # 독립적인 기준 정답
    score_gold = (
        q_stored.astype(np.int64)
        @ K_stored.astype(np.int64).T
    )

    assert planes.shape == (8, 37, 64)
    assert partials.shape == (8, 3, 37)
    assert score_bitplane.shape == (3, 37)

    # 두 방식의 결과가 일치하는지 검증
    np.testing.assert_array_equal(
        score_bitplane,
        score_gold,
    )


def test_cumulative_matches_accumulate_for_every_m():
    """cumulative[m]과 accumulate(partials, m)가 항상 같은지 검증한다."""

    rng = np.random.default_rng(21)

    # 평면 8개, Query 2개, 토큰 5개
    partials = rng.integers(
        -500,
        501,
        size=(8, 2, 5),
        dtype=np.int32,
    )

    cumulative = cumulative_accumulate(partials)

    # cumulative의 shape은 (n_planes+1, n_steps, n_tokens)이어야 한다.
    assert cumulative.shape == (9, 2, 5)

    # 모든 m=0..8에 대해 cumulative[m]과 accumulate(partials, m)가 일치하는지 검증
    for m in range(9):
        np.testing.assert_array_equal(
            cumulative[m],
            accumulate(partials, m=m),
        )

    # 마지막 누산값은 전체 누산값과 같아야 한다.
    np.testing.assert_array_equal(
        cumulative[-1],
        accumulate(partials),
    )


# ---------------------------------------------------------------------------
# 시프트 누산이 자리값과 대응하는가
# ---------------------------------------------------------------------------

def test_accumulate_is_exactly_the_weighted_sum():
    """m=0..8 전부에서 S_m = Σ 2^(7-b)·P_b 인지."""

    rng = np.random.default_rng(0)
    P = rng.integers(-50, 50, size=(8, 4)).astype(np.int64)
    w = plane_weights(8)

    for m in range(9):
        np.testing.assert_array_equal(accumulate(P, m), (w[:m, None] * P[:m]).sum(0))

    # 시프트가 곱셈과 같은지 — 자리값은 2의 거듭제곱이므로 시프트 한 번이다
    for m in range(1, 9):
        shifted = sum(P[b].astype(np.int64) << (7 - b) for b in range(m))
        np.testing.assert_array_equal(accumulate(P, m), shifted)


def test_more_planes_than_given_raises():
    """★ 평면 수보다 큰 m 이 조용히 잘리지 않는지.

    예전에는 m=100 이 m=8 과 같은 값을 냈다. "100장을 처리했다"와
    "다 처리했다"가 구분되지 않으면 상한식의 m 을 잘못 넘겨도 모른다.
    """
    P = np.ones((8, 3), dtype=np.int64)

    with pytest.raises(ValueError, match="only 8 planes"):
        accumulate(P, 9)

    # 음수와 0 은 "아직 아무것도 안 처리" 로 0 을 낸다
    assert not accumulate(P, 0).any()
    assert not accumulate(P, -3).any()


def test_cumulative_stacks_every_s_m():
    """cumulative_accumulate 가 m=0..n 을 한 줄씩 쌓는지."""

    rng = np.random.default_rng(1)
    P = rng.integers(-30, 30, size=(8, 5, 3)).astype(np.int64)
    C = cumulative_accumulate(P)

    assert C.shape == (9,) + P.shape[1:]
    assert not C[0].any()                                   # m=0 은 0
    for m in range(9):
        np.testing.assert_array_equal(C[m], accumulate(P, m))

    # 한 줄 차이가 정확히 그 평면의 기여여야 한다
    for m in range(1, 9):
        np.testing.assert_array_equal(C[m] - C[m - 1], plane_weights(8)[m - 1] * P[m - 1])


# ---------------------------------------------------------------------------
# 스케일 폴딩 — per-channel 과 "곱셈기 없음" 을 동시에
# ---------------------------------------------------------------------------

def test_folding_keeps_per_channel_and_still_has_no_multiplier():
    """★ K 의 채널별 scale 을 q 에 접었을 때 실수 내적이 복원되는지.

    접지 않으면 sc_i 가 토큰 루프 안에 남아 곱셈이 되살아난다. 접으면
    곱셈은 **스텝당 d회**로 끝나고 토큰 수 T 와 무관해진다.
    """
    rng = np.random.default_rng(2)
    d, T, S = 64, 128, 6
    q = rng.normal(0, 1.0, size=(S, d))
    k = rng.normal(0, 1.0, size=(T, d))

    key = quantize_key(k, granularity="per_channel")
    assert np.asarray(key.scale).size == d, "per_channel 이면 차원마다 scale 이 따로다"

    fq = fold_and_quantize_query(q, key)
    got = to_real_scores(exact_int_scores(fq.stored, key), fq, d)
    want = (q @ k.T) / np.sqrt(d)

    rel = np.abs(got - want) / (np.abs(want).std() + 1e-12)
    assert rel.mean() < 0.05, f"mean relative error {rel.mean():.4f}"

    # 접은 뒤의 곱셈 횟수는 T 에 안 붙는다 — q 쪽 모양만 커진다
    assert fq.stored.shape == (S, d)


def test_zero_point_correction_has_no_token_axis():
    """보정항이 토큰마다 다르면 순위가 바뀐다 — 축 자체가 없어야 한다."""

    rng = np.random.default_rng(3)
    d, T, S = 32, 40, 4
    key = quantize_key(rng.normal(0, 1, size=(T, d)))
    fq = fold_and_quantize_query(rng.normal(0, 1, size=(S, d)), key)

    assert fq.zp_correction.shape == (S,), "보정항에 토큰 축이 생기면 상수가 아니다"

    # 정수 점수 순위와 실수 점수 순위가 스텝마다 같아야 한다
    s_int = exact_int_scores(fq.stored, key)
    s_real = to_real_scores(s_int, fq, d)
    for i in range(S):
        np.testing.assert_array_equal(np.argsort(s_int[i]), np.argsort(s_real[i]))


# ---------------------------------------------------------------------------
# 참값 일치와 /sqrt(d)
# ---------------------------------------------------------------------------

def test_exact_int_scores_is_bit_exact_with_the_plane_path():
    """비트평면 경로와 정수 행렬곱이 비트 단위로 같은지."""

    rng = np.random.default_rng(4)
    d, T, S = 64, 96, 5
    key = quantize_key(rng.normal(0, 1, size=(T, d)))
    fq = fold_and_quantize_query(rng.normal(0, 1, size=(S, d)), key)

    s_plane = accumulate(partial_dots(fq.stored, to_bitplanes(key.stored, 8)))
    s_exact = exact_int_scores(fq.stored, key)

    np.testing.assert_array_equal(s_plane, s_exact)
    np.testing.assert_array_equal(
        s_exact, fq.stored.astype(np.int64) @ key.stored.astype(np.int64).T
    )
    assert s_plane.dtype == np.int64 and s_exact.dtype == np.int64


def test_rsqrt_d_comes_after_the_zero_point_correction():
    """/sqrt(d) 의 위치 — 보정을 뺀 뒤에 나눠야 한다."""

    rng = np.random.default_rng(5)
    d, T, S = 64, 50, 3
    key = quantize_key(rng.normal(0, 1, size=(T, d)))
    fq = fold_and_quantize_query(rng.normal(0, 1, size=(S, d)), key)
    s_int = exact_int_scores(fq.stored, key)

    scale = np.asarray(fq.scale, dtype=np.float64).reshape(-1)[:, None]
    manual = (s_int - fq.zp_correction[:, None]) * scale / np.sqrt(d)
    np.testing.assert_array_equal(to_real_scores(s_int, fq, d), manual)

    # 끄면 정확히 sqrt(d) 배 — 나누는 자리가 하나뿐이라는 뜻이다
    np.testing.assert_allclose(
        to_real_scores(s_int, fq, d, apply_rsqrt_d=False), manual * np.sqrt(d)
    )


def test_a_single_step_no_longer_borrows_step_zeros_scale():
    """★ 1차원 입력이 0번 스텝의 보정·스케일을 쓰던 자리.

    to_real_scores(s_int[i], fq, d) 는 i 가 무엇이든 0번 스텝의 zp/scale 을
    썼다. 스텝이 5개면 4개가 조용히 틀린 값이 된다.
    """
    rng = np.random.default_rng(6)
    d, T, S = 32, 40, 5
    key = quantize_key(rng.normal(0, 1, size=(T, d)))
    fq = fold_and_quantize_query(rng.normal(0, 1, size=(S, d)), key)
    s_int = exact_int_scores(fq.stored, key)

    with pytest.raises(ValueError, match="1-D but fq holds 5 steps"):
        to_real_scores(s_int[2], fq, d)

    # 스텝이 하나뿐이면 모호하지 않으므로 그대로 통과한다
    fq1 = fold_and_quantize_query(rng.normal(0, 1, size=(1, d)), key)
    one = exact_int_scores(fq1.stored, key)
    np.testing.assert_array_equal(
        to_real_scores(one[0], fq1, d), to_real_scores(one, fq1, d)[0]
    )
