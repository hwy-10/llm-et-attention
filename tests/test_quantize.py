"""양자화 · 비트평면 규약 검증.

여기가 틀리면 이후 전부 무의미하다.
"""

import numpy as np
import pytest

from src.config import load_config
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
    assert np.all(w > 0), "a negative K weight invalidates the early-termination upper bound"
    assert w.tolist() == [128, 64, 32, 16, 8, 4, 2, 1], "weights must be in MSB-first order"


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
    assert rel.mean() < 0.05, f"mean relative error {rel.mean():.4f} after zero-point correction is too large"


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


# ---------------------------------------------------------------------------
# 왕복 · 순서 · 범위 — 여기가 5.4절 상한식의 전제다
# ---------------------------------------------------------------------------

def test_bitplane_roundtrip_over_a_thousand_cases():
    """왕복이 무손실인지 무작위 1000케이스로 확인."""

    rng = np.random.default_rng(0)
    for _ in range(1000):
        n, d = int(rng.integers(1, 40)), int(rng.integers(1, 40))
        s = rng.integers(0, 256, size=(n, d), dtype=np.uint16)
        np.testing.assert_array_equal(from_bitplanes(to_bitplanes(s, 8)), s.astype(np.int64))


def test_values_that_do_not_fit_are_rejected():
    """★ 담기지 않는 값이 조용히 잘리지 않는지.

    예전에는 256 이 0 으로, -1 이 255 로 바뀌어 왕복이 실패하는데도
    아무 말이 없었다. 무손실 구간은 0..2^n-1 뿐이다.
    """
    for bad in (256, 65535, -1):
        with pytest.raises(ValueError, match=r"must be in \[0, 255\]"):
            to_bitplanes(np.array([[bad]], dtype=np.int64), 8)

    # 경계값은 통과해야 한다
    for good in (0, 255):
        got = from_bitplanes(to_bitplanes(np.array([[good]], dtype=np.int64), 8))
        assert int(got[0, 0]) == good


def test_plane_zero_is_the_msb():
    """★ MSB 우선 순서 — to_bitplanes 의 index 0 이 b7 인지 직접 고정.

    왕복 검사로는 이것을 못 잡는다. to_bitplanes 와 plane_weights 를 **둘 다**
    LSB 우선으로 뒤집으면 왕복은 그대로 통과하는데, 평면 순서가 뒤집혀
    조기 종단의 전제(위 자리부터 확정된다)가 통째로 무너진다.
    """
    assert to_bitplanes(np.array([[128]], dtype=np.int64), 8)[:, 0, 0].tolist() == [1] + [0] * 7
    assert to_bitplanes(np.array([[1]], dtype=np.int64), 8)[:, 0, 0].tolist() == [0] * 7 + [1]

    # 평면 t 의 자리값은 2^(7-t) 여야 한다 — 자리값표와 평면 순서가 같은 규약인지
    for t in range(8):
        one = np.zeros((8, 1, 1), dtype=np.uint8)
        one[t] = 1
        assert int(from_bitplanes(one)[0, 0]) == 1 << (7 - t)
        assert int(plane_weights(8)[t]) == 1 << (7 - t)


def test_remaining_scale_for_every_m():
    """m=0..8 전부와 범위 밖."""

    for m in range(9):
        assert remaining_scale(m, 8) == (1 << (8 - m)) - 1

    # m=-1 은 511(평면 9개분)을 조용히 냈고, m=9 는 시프트 오류만 냈다
    for m in (-1, 9):
        with pytest.raises(ValueError, match="must be in"):
            remaining_scale(m, 8)


# ---------------------------------------------------------------------------
# 극단 입력 탐색 — 못 찾았으면 못 찾았다고 적는다
# ---------------------------------------------------------------------------

def test_quantize_key_stays_in_range_for_extreme_inputs():
    """0..255 를 벗어나는 입력을 찾으려 시도한 기록.

    아래 8종 어디서도 벗어나지 않았다. clip 이 마지막에 한 번 더 걸리기 때문이다.
    """
    rng = np.random.default_rng(1)
    cases = {
        "정규분포":      rng.normal(0, 1, size=(64, 8)),
        "전부 같은 값":   np.full((64, 8), 5.0),
        "한 채널만 상수": np.concatenate([np.full((64, 1), 3.0), rng.normal(0, 1, (64, 7))], axis=1),
        "거대값":        rng.normal(0, 1e30, size=(64, 8)),
        "미세값":        rng.normal(0, 1e-30, size=(64, 8)),
        "이상치 하나":    np.r_[rng.normal(0, 1, (63, 8)), np.full((1, 8), 1e12)],
        "클리핑 99%":    rng.standard_cauchy(size=(64, 8)),
        "1 행":          rng.normal(0, 1, size=(1, 8)),
    }
    for name, k in cases.items():
        pct = 99.0 if name == "클리핑 99%" else 100.0
        st = quantize_key(k, clip_percentile=pct).stored
        assert st.dtype == np.uint8 and st.min() >= 0 and st.max() <= 255, name


def test_non_finite_input_is_rejected():
    """★ NaN / inf 는 범위 검사를 통과하면서 값만 사라진다.

    clip(nan, 0, 255) 이 nan 이고 uint8 로 바뀌면서 0 이 된다. stored 는
    0..255 안에 얌전히 들어와 있으므로 범위만 보는 검사로는 절대 못 잡는다.
    """
    rng = np.random.default_rng(2)
    for bad in (np.nan, np.inf, -np.inf):
        k = rng.normal(0, 1, size=(16, 4))
        k[3, 2] = bad
        with pytest.raises(ValueError, match="non-finite"):
            quantize_key(k)


# ---------------------------------------------------------------------------
# config/quant.yaml 과 코드 대조
# ---------------------------------------------------------------------------

def test_quant_yaml_matches_the_code():
    """★ quant.yaml 의 9개 키를 전수 대조.

    이 파일은 "함부로 바꾸면 안 된다"고 적혀 있는데 실제로 코드가 읽는 것은
    3개뿐이다. 나머지 6개는 규약을 적어 둔 것이라 바꿔도 조용하다.
    바뀌면 여기서 걸리도록 코드가 실제로 하는 일과 묶어 둔다.
    """
    import inspect

    cfg = load_config()
    planes, key, query = cfg.get("quant.planes"), cfg.get("quant.key"), cfg.get("quant.query")

    # (1) 코드가 읽는 3개
    assert cfg.n_planes == int(planes["n_planes"])
    sig = inspect.signature(quantize_key).parameters
    assert key["granularity"] == sig["granularity"].default
    assert float(key["clip_percentile"]) == sig["clip_percentile"].default

    # (2) 읽지 않는 6개 — 코드가 하드코딩한 값과 같아야 한다
    #     key.bits: decode_loop 가 bits 자리에 n_planes 를 넘긴다. 다르면
    #     to_bitplanes 가 엉뚱한 비트를 뽑으므로 둘은 애초에 갈라질 수 없다.
    assert int(key["bits"]) == int(planes["n_planes"])
    assert int(query["bits"]) == inspect.signature(quantize_query).parameters["bits"].default
    assert query["granularity"] == inspect.signature(quantize_query).parameters["granularity"].default

    # scheme / order 는 값이 아니라 규약이다. 코드가 지키는 성질로 고정한다.
    assert key["scheme"] == "asymmetric_unsigned"
    assert np.all(plane_weights(cfg.n_planes) > 0)          # unsigned 라 자리값이 전부 양수

    assert query["scheme"] == "symmetric_signed"
    st = quantize_query(np.array([[-1.0, 1.0]])).stored
    assert st.min() >= -127 and st.max() <= 127             # -128 을 안 쓰는 대칭 범위

    assert planes["order"] == "msb_first"
    assert to_bitplanes(np.array([[128]], dtype=np.int64), 8)[0, 0, 0] == 1
# ---------------------------------------------------------------------------
# 분해가 감당하는 범위 — 차원 · 평면 수 · 입력 타입
# ---------------------------------------------------------------------------

def test_any_rank_decomposes_to_the_same_shape():
    """★ 1차원 입력이 없는 축을 만들어 내던 자리.

    예전에는 shifts 를 (n, 1, 1) 로 고정해 2차원만 맞았다. 1차원을 넣으면
    예외도 안 나고 (8, 1, 3) 이 나왔다 — 있지도 않은 축이 하나 생긴 것이다.
    """
    rng = np.random.default_rng(0)
    for shape in ((3,), (2, 3), (2, 3, 4), (2, 2, 2, 2)):
        a = rng.integers(0, 256, size=shape, dtype=np.int64)
        planes = to_bitplanes(a, 8)
        assert planes.shape == (8,) + shape, shape
        np.testing.assert_array_equal(from_bitplanes(planes), a)


def test_wide_planes_survive_the_round_trip():
    """★ 범위 가드가 허용하는 값을 구현이 감당하는지.

    가드는 0 .. 2^n_planes-1 을 허용한다고 말하는데 예전 구현은 uint16 으로
    캐스트해 65535 에서 잘랐다. n_planes=17 에서 131,071 을 넣으면 가드는
    통과하고 값만 65,535 로 사라졌다.
    """
    for n in (8, 16, 17, 20, 32):
        limit = (1 << n) - 1
        a = np.array([[limit, 0, 1]], dtype=np.int64)
        got = from_bitplanes(to_bitplanes(a, n))
        np.testing.assert_array_equal(got, a), n

    # 가드가 말하는 상한과 실제로 되돌아오는 상한이 같아야 한다
    with pytest.raises(ValueError, match="must be in"):
        to_bitplanes(np.array([[1 << 17]], dtype=np.int64), 17)


def test_a_float_array_is_rejected():
    """실수는 예전에 uint16 캐스트가 소수점을 조용히 버렸다 (3.7 -> 3)."""

    with pytest.raises(ValueError, match="integer array"):
        to_bitplanes(np.array([[3.7, 1.2]]))

    # 정수형이면 dtype 이 무엇이든 통과한다
    for dt in (np.uint8, np.uint16, np.int32, np.int64):
        a = np.array([[200, 32]], dtype=dt)
        np.testing.assert_array_equal(from_bitplanes(to_bitplanes(a, 8)), a)


def test_from_bitplanes_keeps_the_wide_type_by_default():
    """★ 좁은 dtype 을 기본으로 두면 n_planes > 8 에서 예외 없이 잘린다."""

    a = np.array([[4095]], dtype=np.int64)          # 12비트
    planes = to_bitplanes(a, 12)

    assert from_bitplanes(planes).dtype == np.int64
    assert int(from_bitplanes(planes)[0, 0]) == 4095

    # 좁게 달라고 하면 달라는 대로 준다 — 다만 그건 호출자가 고른 것이다
    assert int(from_bitplanes(planes, np.uint8)[0, 0]) == 255
