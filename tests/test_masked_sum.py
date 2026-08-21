import numpy as np
import pytest

from src.masked_sum import (
    AdderTreeModel,
    BaselineMacModel,
    accumulator_bits,
    partial_dot_single,
    partial_dots,
)


def test_partial_dot_single_manual():
    q = np.array([2, -3, 5, 1], dtype=np.int16)
    """임시 Query 하나입니다. 실제 프로젝트에서는 64차원이지만, 손으로 확인하기 쉽게 4차원으로 줄였습니다."""

    k_plane = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
    ], dtype=np.uint8)

    got = partial_dot_single(q, k_plane)
    expected = np.array([7, -2, -1], dtype=np.int32)

    np.testing.assert_array_equal(got, expected)
    """Query와 Key 비트평면 하나를 입력했을 때, Key 비트가 1인 위치의 Query 값만 정확히 더하는지 검증했습니다."""


def test_partial_dots_manual():
    q = np.array([
        [2, -3, 5, 1],
        [-1, 4, 2, -2],
    ], dtype=np.int16)
    """Query 2개입니다. 실제 프로젝트에서는 64차원이지만, 손으로 확인하기 쉽게 4차원으로 줄였습니다."""

    k_planes = np.array([
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ],
        [
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
    ], dtype=np.uint8)

    got = partial_dots(q, k_planes)

    expected = np.array([
        [
            [7, -2, -1],
            [1, 2, 3],
        ],
        [
            [6, 3, 2],
            [0, -3, 6],
        ],
    ], dtype=np.int32)

    assert got.shape == (2, 2, 3)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, expected)
    """Query 2개와 Key 비트평면 2개를 입력했을 때, 모든 조합의 부분 내적이 정확히 계산되는지 검증했습니다."""


def test_chunk_result_is_identical():
    rng = np.random.default_rng(10)

    q = rng.integers(
        -127, 128,
        size=(3, 64),
        dtype=np.int16,
    )

    k_planes = rng.integers(
        0, 2,
        size=(8, 37, 64),
        dtype=np.uint8,
    )

    whole = partial_dots(q, k_planes, chunk=0)
    chunked = partial_dots(q, k_planes, chunk=7)

    np.testing.assert_array_equal(whole, chunked)
    """메모리 사용을 줄이기 위해 청크를 나누어 계산하였고 그 결과가 통째로 계산한 결과와 동일한지 검증했습니다."""


def test_single_query_is_accepted():
    q = np.array([2, -3, 5, 1], dtype=np.int16)

    k_planes = np.array([
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
    ], dtype=np.uint8)

    got = partial_dots(q, k_planes)

    assert got.shape == (1, 1, 2)
    np.testing.assert_array_equal(
        got,
        np.array([[[7, -2]]], dtype=np.int32),
    )
    """단일 Query와 Key 비트평면을 입력했을 때, 부분 내적이 정확히 계산되는지 검증했습니다."""


def test_head_dim_mismatch_raises_error():
    q = np.zeros((1, 4), dtype=np.int16)
    k_planes = np.zeros((8, 3, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        partial_dots(q, k_planes)
        """Query와 Key 비트평면의 head_dim이 일치하지 않을 때 ValueError가 발생하는지 검증했습니다."""
        

def test_hardware_model_default_values():
    model = AdderTreeModel(
        n_inputs=64,
        input_bits=8,
    )

    summary = model.summary()

    # 트리 구조에서 유도되는 값 — 바뀌면 실제로 잘못된 것이다.
    assert summary["depth"] == 6
    assert summary["n_adders"] == 63
    assert summary["output_bits"] == 14
    assert summary["uses_dsp"] is False

    # 가산 트리의 총 가산 비트 수 624 도 구조에서 유도된다.
    # 단별로 (가산기 수 x 가산기 폭) 을 더한 값: 288+160+88+48+26+14 = 624.
    # lut_per_fa 로 나눠 계수와 무관하게 비교한다.
    assert model.est_lut / model.lut_per_fa == 624

    # est_lut_* 자체는 Vivado 실측으로 교체될 1차 추정이다 (masked_sum.py docstring).
    # lut_per_fa=1.0 과 "6-LUT 가 2비트 처리" 는 가정이므로 값을 못박지 않고
    # 성질만 본다 — 못박으면 실측 반영이 곧 테스트 실패가 된다.
    assert summary["est_lut_mask"] > 0
    assert summary["est_lut_total"] > summary["est_lut_tree"]
    """head_dim=64, Query 8비트 조건에서 가산 트리의 깊이·가산기 수·출력 비트폭·총 가산 비트가
    예상한 값을 반환하는지 검증했습니다. LUT 추정치는 가정에 의존하므로 성질만 확인합니다."""


def test_baseline_and_accumulator_width():
    baseline = BaselineMacModel(
        n_pe=32,
        dsp_per_mac=1,
    )

    assert baseline.est_dsp == 32
    assert baseline.summary()["uses_dsp"] is True
    assert accumulator_bits(head_dim=64) == 22
    """32개 PE, 1 DSP/PE 조건에서 기준 설계의 DSP 수와 누산기 비트폭이 예상한 값을 반환하는지 검증했습니다."""
