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


def test_partial_dots_is_select_and_add_not_multiply():
    """★ "곱셈이 없다" 를 의미로 못박는다 — 구현은 einsum 이어도 좋다.

    einsum 은 빠르지만 코드만 보면 곱셈으로 읽힌다. 그래서 "비트가 1이면 q 를 고르고
    0이면 0을 고른 뒤 더한다" 는 정의를 따로 계산해 두 값이 같은지 본다.

    구현을 마스킹으로 바꾸지 않은 이유는 규모다. 실제 워크로드(8평면 x 480스텝 x
    512토큰 x 64차원)에서 마스킹 판은 중간 배열이 **0.5 GB** 로 지금(7.9 MB)의 64배가
    되고 4배 느리다. 값은 같으므로 빠른 쪽을 두고 의미를 검사로 지킨다.
    """
    rng = np.random.default_rng(0)
    for shape in ((3, 5, 4), (1, 1, 8), (8, 7, 16)):
        n_planes, n_tokens, head_dim = shape
        q = rng.integers(-127, 128, size=(4, head_dim)).astype(np.int32)
        kp = rng.integers(0, 2, size=shape).astype(np.uint8)

        # 정의 그대로: 비트가 1인 자리의 q 만 골라 더한다. 곱셈이 한 번도 안 나온다
        select_add = np.where(kp[:, None, :, :] == 1, q[None, :, None, :], 0)
        want = select_add.sum(axis=-1, dtype=np.int32)

        np.testing.assert_array_equal(partial_dots(q, kp), want)

    # 비트가 0/1 이 아니면 위 동치가 깨진다 — 전제를 같이 못박는다
    kp2 = np.array([[[2, 0]]], dtype=np.uint8)
    q2 = np.array([[3, 5]], dtype=np.int32)
    assert int(partial_dots(q2, kp2)[0, 0, 0]) == 6      # einsum 은 곱한다
    assert int(np.where(kp2[:, None] == 1, q2[None, :, None], 0).sum()) == 0


def test_accumulator_width_holds_the_true_worst_case():
    """★ 누산기 폭이 최악값을 실제로 담는지 — 식이 아니라 범위로 확인.

    q 의 최악값은 +127 이 아니라 -128 이다. 폭 계산이 그쪽을 빠뜨리면 조용히 한 비트
    모자란 값이 나온다.
    """
    # q_bits 가 작을수록 -2^(b-1) 과 +2^(b-1)-1 의 차이가 커진다.
    # 2 를 빼면 두 식이 같은 값을 내 이 검사가 판별력을 잃는다 (실제로 그랬다).
    for head_dim in (1, 8, 32, 64, 128):
        for q_bits in (2, 3, 4, 8, 12):
            for n_planes in (1, 4, 8, 12):
                bits = accumulator_bits(head_dim, q_bits=q_bits, n_planes=n_planes)

                k_max = (1 << n_planes) - 1          # unsigned K 의 최댓값
                lo = -k_max * (1 << (q_bits - 1)) * head_dim     # q = -2^(b-1)
                hi = k_max * ((1 << (q_bits - 1)) - 1) * head_dim

                assert -(1 << (bits - 1)) <= lo, (head_dim, q_bits, n_planes, bits, lo)
                assert hi <= (1 << (bits - 1)) - 1, (head_dim, q_bits, n_planes, bits, hi)

                # 헐렁하지도 않아야 한다. 다만 식이 log2 올림이라 모서리에서 1비트
                # 여유가 생기므로(head_dim=1, q_bits=2) 최소폭+1 까지만 허용한다.
                minimal = next(b for b in range(2, 128)
                               if -(1 << (b - 1)) <= lo and hi <= (1 << (b - 1)) - 1)
                assert bits <= minimal + 1, (head_dim, q_bits, n_planes, bits, minimal)


def test_adder_tree_pipelined_latency_is_the_depth():
    """완전 파이프라인 지연 = 트리 깊이. head_dim 64 에서 6 사이클."""
    assert AdderTreeModel(n_inputs=64).fully_pipelined_latency_cycles == 6
    for n in (2, 4, 8, 16, 32, 64, 128):
        m = AdderTreeModel(n_inputs=n)
        assert m.fully_pipelined_latency_cycles == m.depth
