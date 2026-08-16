"""양자화 · 비트평면 규약 검증.

여기가 틀리면 이후 전부 무의미하다.
"""

import numpy as np

from src.accumulator import exact_int_scores, fold_and_quantize_query, to_real_scores
from src.quantize import (
    from_bitplanes,
    plane_weights,
    quantize_key,
    quantize_query,
    remaining_scale,
    to_bitplanes,
)


def test_plane_weights_all_positive():
    """★ 상한식 R_m = (2^(8-m)-1)·Q+ 성립의 전제 ★

    K 를 unsigned 로 저장하므로 모든 자리값이 양수여야 한다.
    2의 보수(signed)로 바꾸면 MSB 가 -128 이 되어 상한식이 깨진다.
    """
    w = plane_weights(8)
    assert np.all(w > 0), "K 자리값에 음수가 있으면 조기 종단 상한이 무효가 된다"
    assert w.tolist() == [128, 64, 32, 16, 8, 4, 2, 1], "MSB 우선 순서여야 한다"


def test_bitplane_roundtrip():
    rng = np.random.default_rng(0)
    stored = rng.integers(0, 256, size=(37, 64), dtype=np.uint16)
    planes = to_bitplanes(stored, 8)
    assert planes.shape == (8, 37, 64)
    assert set(np.unique(planes).tolist()) <= {0, 1}
    np.testing.assert_array_equal(from_bitplanes(planes), stored.astype(np.int64))


def test_remaining_scale():
    # m 개 처리 후 남은 자리값의 합
    assert remaining_scale(0, 8) == 255
    assert remaining_scale(1, 8) == 127
    assert remaining_scale(4, 8) == 15
    assert remaining_scale(8, 8) == 0


def test_key_quant_range_and_error():
    rng = np.random.default_rng(1)
    k = rng.normal(0, 1.0, size=(128, 64))
    kq = quantize_key(k, granularity="per_channel")
    assert kq.stored.dtype == np.uint8
    assert kq.stored.min() >= 0 and kq.stored.max() <= 255
    err = np.abs(kq.dequantize() - k)
    # 채널별 min/max 양자화이므로 오차는 스텝의 절반 이하
    assert np.all(err <= (np.asarray(kq.scale).reshape(1, -1) * 0.5 + 1e-9))


def test_query_quant_symmetric():
    rng = np.random.default_rng(2)
    q = rng.normal(0, 1.0, size=(16, 64))
    qq = quantize_query(q)
    assert qq.stored.min() >= -127 and qq.stored.max() <= 127
    # -128 은 쓰지 않는다 (대칭 범위)
    assert not np.any(qq.stored == -128)


def test_zero_point_correction_identity():
    """★ zero-point 보정항이 실제 내적을 복원하는지 ★

    s_real = scale_q · ( q_st·K_st − Σ q_st·z )  /  sqrt(d)
    가 실수 내적 q·k/sqrt(d) 를 (양자화 오차 안에서) 복원해야 한다.
    """
    rng = np.random.default_rng(3)
    d, T, S = 64, 96, 8
    q = rng.normal(0, 1.0, size=(S, d))
    k = rng.normal(0, 1.0, size=(T, d))

    key = quantize_key(k, granularity="per_channel")
    fq = fold_and_quantize_query(q, key)
    s_int = exact_int_scores(fq.stored, key)
    got = to_real_scores(s_int, fq, d)

    want = (q @ k.T) / np.sqrt(d)
    rel = np.abs(got - want) / (np.abs(want).std() + 1e-12)
    assert rel.mean() < 0.05, f"zero-point 보정 후 평균 상대오차 {rel.mean():.4f} 가 너무 크다"


def test_zero_point_correction_is_rank_invariant():
    """보정항은 모든 토큰에 동일한 상수이므로 순위를 바꾸지 않는다.

    -> 종단 판정은 정수 s_int 만 보면 된다 (하드웨어 단순화의 근거).
    """
    rng = np.random.default_rng(4)
    d, T = 64, 64
    q = rng.normal(0, 1.0, size=(1, d))
    k = rng.normal(0, 1.0, size=(T, d))
    key = quantize_key(k)
    fq = fold_and_quantize_query(q, key)
    s_int = exact_int_scores(fq.stored, key)[0]
    s_real = to_real_scores(s_int, fq, d)
    np.testing.assert_array_equal(np.argsort(s_int), np.argsort(s_real))
