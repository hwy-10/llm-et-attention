"""손익분기 계산 검증 — 배경지식 가이드 6.3-(4)

이 파일의 수식이 프로젝트의 결론이다.

    실효 speedup = (cycles_기준 / cycles_제안) x (fmax_제안 / fmax_기준)

사이클만 보면 안 된다는 것이 요점이므로, 두 인자가 각각 제 자리에
들어가는지를 손 계산으로 고정한다.
"""

import warnings

import numpy as np
import pytest

from src.config import ConfigDefaultWarning, load_config
from utils.cost_model import (
    ResourceReport,
    breakeven_curve,
    compare,
    control_overhead,
    effective_speedup,
    energy_proxy_uj,
    latency_us,
    max_tolerable_derate,
)


# ---------------------------------------------------------------------------
# 수식
# ---------------------------------------------------------------------------

def test_effective_speedup_by_hand():
    """두 인자가 각각 제 자리에 들어가는지."""

    # 사이클이 절반이고 주파수가 그대로면 정확히 2배
    assert effective_speedup(1000, 200.0, 500, 200.0) == pytest.approx(2.0)

    # 사이클이 그대로이고 주파수가 절반이면 정확히 0.5배
    assert effective_speedup(1000, 200.0, 1000, 100.0) == pytest.approx(0.5)

    # 사이클 25% 절감(0.75배) + Fmax 20% 하락(0.8배) -> 1.0667
    assert effective_speedup(1000, 200.0, 750, 160.0) == pytest.approx(4 / 3 * 0.8)

    # 두 인자를 바꿔 넣으면 다른 값이 나온다 — 자리 바뀜을 잡는다
    assert effective_speedup(1000, 200.0, 750, 160.0) != pytest.approx(
        effective_speedup(750, 160.0, 1000, 200.0)
    )


def test_speedup_refuses_to_return_zero_silently():
    """0.0 은 '무한히 느리다' 와 구분이 안 된다."""

    with pytest.raises(ValueError, match="cycles_new"):
        effective_speedup(1000, 200.0, 0, 200.0)

    # 자원 섹션을 못 읽으면 fmax 가 0 이 된다. 그 상태로 결론을 내면 안 된다.
    with pytest.raises(ValueError, match="fmax_ref"):
        effective_speedup(1000, 0.0, 500, 200.0)


def test_latency_and_energy_units():
    """단위 — 사이클/MHz = us, mW x us = nJ 이므로 uJ 는 x 1e-3."""

    # 200 사이클을 200 MHz 로 -> 1 us
    assert latency_us(200, 200.0) == pytest.approx(1.0)

    # 100 mW 로 1 us -> 100 nJ = 0.1 uJ
    assert energy_proxy_uj(200, 200.0, 100.0) == pytest.approx(0.1)

    # 주파수를 모르면 무한대 — 0 으로 삼키지 않는다
    assert latency_us(200, 0.0) == float("inf")


def test_max_tolerable_derate_is_exactly_the_breakeven():
    """허용 Fmax비를 그대로 넣으면 실효 speedup 이 정확히 1.0 이어야 한다."""

    for c_ref, c_new in ((1000, 750), (1000, 1000), (1000, 1200)):
        ratio = max_tolerable_derate(c_ref, c_new)
        assert ratio == pytest.approx(c_new / c_ref)

        sp = effective_speedup(c_ref, 200.0, c_new, 200.0 * ratio)
        assert sp == pytest.approx(1.0)


def test_breakeven_curve_matches_the_scalar_formula():
    """격자 값이 스칼라 수식과 같은지."""

    cycles = [600.0, 800.0, 1000.0]
    ratios = [0.8, 0.9, 1.0]
    grid = breakeven_curve(1000.0, cycles, ratios)

    assert grid.shape == (3, 3)
    for i, cn in enumerate(cycles):
        for j, fr in enumerate(ratios):
            assert grid[i, j] == pytest.approx(
                effective_speedup(1000.0, 200.0, cn, 200.0 * fr)
            )

    # 0 이 섞이면 inf 가 격자에 퍼진다
    with pytest.raises(ValueError, match="non-positive"):
        breakeven_curve(1000.0, [600.0, 0.0], ratios)


# ---------------------------------------------------------------------------
# 설정 배선
# ---------------------------------------------------------------------------

def test_resource_report_warns_when_the_section_is_missing():
    """섹션이 없으면 전부 0 이 되어 speedup 0.0 / latency inf 로 흘러간다."""

    cfg = load_config()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for key in ("baseline", "seq_no_et", "exact_et", "approx_et"):
            r = ResourceReport.from_config(cfg, key)
            assert r.name == key
            assert r.fmax_mhz > 0
    assert not [w for w in caught if issubclass(w.category, ConfigDefaultWarning)]

    with pytest.warns(ConfigDefaultWarning, match="hardware.resources.nope"):
        blind = ResourceReport.from_config(cfg, "nope")
    assert blind.fmax_mhz == 0.0


# ---------------------------------------------------------------------------
# exp6 의 결론
# ---------------------------------------------------------------------------

def test_the_breakeven_conclusion_decomposes():
    """★ '② 대비 0.93' 이 두 인자의 곱임을 고정한다.

    사이클을 4.3% 줄였는데 Fmax 가 10.8% 떨어져 손해가 되는 구조다.
    """
    cfg = load_config()
    seq = ResourceReport.from_config(cfg, "seq_no_et")
    exa = ResourceReport.from_config(cfg, "exact_et")

    # exp6 (N=512, exact, derate=1.0) 의 사이클
    c_ref, c_new = 34_440.0, 32_969.0

    cycle_ratio = c_new / c_ref
    fmax_ratio = exa.fmax_mhz / seq.fmax_mhz
    sp = effective_speedup(c_ref, seq.fmax_mhz, c_new, exa.fmax_mhz)

    assert cycle_ratio == pytest.approx(0.9573, abs=5e-4)
    assert fmax_ratio == pytest.approx(0.8919, abs=5e-4)
    assert sp == pytest.approx(fmax_ratio / cycle_ratio)
    assert sp == pytest.approx(0.932, abs=1e-3)

    # 손익분기 조건은 Fmax비 >= 사이클비 다. 지금은 모자란다.
    assert fmax_ratio < max_tolerable_derate(c_ref, c_new)


def test_the_conclusion_rests_on_estimated_fmax():
    """★ 결론을 정하는 두 값이 아직 추정치임을 감시.

    Vivado 실측이 들어오면 이 테스트가 실패한다 — 그때 결론을 다시 볼 것.
    """
    cfg = load_config()
    seq = ResourceReport.from_config(cfg, "seq_no_et")
    exa = ResourceReport.from_config(cfg, "exact_et")

    assert seq.source == "estimate", "실측이 들어왔다. exp6 결론을 다시 볼 것"
    assert exa.source == "estimate", "실측이 들어왔다. exp6 결론을 다시 볼 것"

    # 결론이 뒤집히는 지점 — exact_et 의 Fmax 가 이보다 높으면 이득이다
    c_ref, c_new = 34_440.0, 32_969.0
    flip = seq.fmax_mhz * max_tolerable_derate(c_ref, c_new)
    assert 170.0 < flip < 180.0, flip
    assert exa.fmax_mhz < flip


def test_bitstopper_comparison_uses_power_not_area():
    """★ [4] 의 6.9% 는 area 가 아니라 power 다 (README 정정)."""

    cfg = load_config()
    seq = ResourceReport.from_config(cfg, "seq_no_et")
    exa = ResourceReport.from_config(cfg, "exact_et")
    out = control_overhead(seq, exa)

    # 두 축이 부호까지 다르다 — 잘못 고르면 비교가 통째로 틀린다
    assert out["control_lut_overhead"] > 0.3       # 면적은 30% 넘게 는다
    assert out["control_power_overhead"] < 0.0     # 전력은 오히려 준다 (읽기 감소)

    assert out["fmax_drop"] == pytest.approx(1.0 - exa.fmax_mhz / seq.fmax_mhz)


def test_compare_bundles_cycles_resources_and_frequency():
    """compare() 가 세 축을 한 번에 묶는지."""

    ref = ResourceReport(name="a", lut=1000, dsp=32, fmax_mhz=200.0, dynamic_power_mw=100.0)
    new = ResourceReport(name="b", lut=1500, dsp=0, fmax_mhz=160.0, dynamic_power_mw=90.0)
    out = compare(ref, new, 1000.0, 750.0)

    assert out["cycle_ratio"] == pytest.approx(0.75)
    assert out["fmax_ratio"] == pytest.approx(0.8)
    assert out["effective_speedup"] == pytest.approx(4 / 3 * 0.8)
    assert out["is_win"] is True

    # DSP -> LUT 자원 교환이 보여야 한다
    assert out["dsp_saved"] == 32
    assert out["d_lut"] == 500

    # derate 는 Fmax 만 낮춘다. 자원과 허용 Fmax비는 그대로다.
    derated = compare(ref, new, 1000.0, 750.0, fmax_derate=0.9)
    assert derated["fmax_new_mhz"] == pytest.approx(144.0)
    assert derated["d_lut"] == out["d_lut"]
    assert derated["max_tolerable_fmax_ratio"] == out["max_tolerable_fmax_ratio"]
    assert derated["effective_speedup"] < out["effective_speedup"]


def test_energy_ratio_uses_the_derated_frequency():
    """주파수가 낮아지면 시간이 늘어 에너지가 는다."""

    ref = ResourceReport(name="a", fmax_mhz=200.0, dynamic_power_mw=100.0)
    new = ResourceReport(name="b", fmax_mhz=200.0, dynamic_power_mw=100.0)

    same = compare(ref, new, 1000.0, 1000.0)
    slow = compare(ref, new, 1000.0, 1000.0, fmax_derate=0.5)

    assert same["energy_ratio"] == pytest.approx(1.0)
    assert slow["energy_ratio"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# exp6 의 배선
# ---------------------------------------------------------------------------

def test_exp6_maps_every_design_to_a_real_resource_section():
    """설계 -> 자원 섹션 이름이 config 에 실제로 있는지."""

    from experiments.exp6_breakeven import _DESIGN_TO_RESOURCE

    cfg = load_config()
    for design, key in _DESIGN_TO_RESOURCE.items():
        section = cfg.get(f"hardware.resources.{key}")
        assert isinstance(section, dict) and section, f"{design} -> {key}"


def test_exp6_still_measures_the_compute_axis():
    """★ exp6 가 어느 사이클 축으로 결론을 내는지 고정한다.

    연산만 세면 조기 종단의 이득(읽기 감소)이 축에 안 나타난다.
    바꾸려면 이 테스트를 함께 고쳐야 하므로 조용히 넘어가지 않는다.
    """
    import inspect

    from experiments import exp6_breakeven

    src = inspect.getsource(exp6_breakeven.run)
    assert '"total_cycles"' in src
    assert "total_cycles_with_memory" not in src, (
        "exp6 가 메모리 포함 축으로 바뀌었다 — 논문 수치를 다시 볼 것"
    )


def test_the_derate_sweep_crosses_the_breakeven_line():
    """★ 스윕이 손익분기선을 지나야 '얼마나 견고한가' 에 답할 수 있다.

    아래쪽만 쓸면 전부 손해로 나와 어디서 뒤집히는지 알 수 없다.
    """
    cfg = load_config()
    sweep = (cfg.get("sweeps.exp6", {}) or {}).get("fmax_derate_sweep", [])

    assert sweep, "fmax_derate_sweep 이 비어 있다"
    assert max(sweep) > 1.0, f"낙관 시나리오가 없다: {sweep}"
    assert min(sweep) < 1.0, f"비관 시나리오가 없다: {sweep}"

    # N=512 exact 는 derate 1.07 부근에서 뒤집힌다 — 스윕이 그 지점을 감싸야 한다
    seq = ResourceReport.from_config(cfg, "seq_no_et")
    exa = ResourceReport.from_config(cfg, "exact_et")
    flip = (32_969.0 / 34_440.0) * seq.fmax_mhz / exa.fmax_mhz

    assert min(sweep) < flip < max(sweep), f"뒤집히는 지점 {flip:.3f} 이 스윕 밖이다"
