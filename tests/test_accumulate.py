"""시프트 누산기 검증.

검증 대상
1. 평면별 가중치 누산
2. 단계별 누산값
3. 비트평면 방식과 일반 INT8 내적의 일치
4. 64차원 무작위 입력
"""

import numpy as np

from src.accumulator import (
    accumulate,
    cumulative_accumulate,
    exact_int_scores,
)
from src.masked_sum import partial_dots
from src.quantize import KeyQuant, to_bitplanes


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