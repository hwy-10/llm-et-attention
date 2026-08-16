"""Vivado 보고서 / 시뮬레이션 CSV 파싱 검증 (mock 데이터 사용)."""

import pytest

from src.config import PROJECT_ROOT
from utils.crosscheck import compare, report
from utils.hw_parser import load_report, parse_sim_csv, parse_timing, parse_utilization

MOCK = PROJECT_ROOT / "rtl_data" / "mock"


def test_parse_utilization():
    d = parse_utilization(MOCK / "exact_et_synth.rpt")
    assert d["lut"] == 15600
    assert d["ff"] == 8900
    assert d["dsp"] == 0
    assert d["bram36"] == 16.0


def test_parse_timing_fmax():
    d = parse_timing(MOCK / "exact_et_synth.rpt")
    assert d["wns_ns"] == pytest.approx(0.340)
    assert d["clock_period_ns"] == pytest.approx(6.0)
    # Fmax = 1000 / (6.0 - 0.34) = 176.68 MHz
    assert d["fmax_mhz"] == pytest.approx(1000.0 / 5.66, rel=1e-6)


def test_load_report_merges_all():
    d = load_report(MOCK / "exact_et_synth.rpt")
    assert {"lut", "ff", "dsp", "fmax_mhz", "dynamic_power_mw"} <= set(d)
    assert d["dynamic_power_mw"] == pytest.approx(168.0)


def test_parse_sim_csv():
    rows = parse_sim_csv(MOCK / "rtl_simulation_cycles.csv")
    assert len(rows) == 12
    assert rows[0]["design"] == "baseline"
    assert isinstance(rows[0]["cycles"], int)


def test_missing_column_raises_loudly(tmp_path):
    """★ 조용한 실패를 막는 장치 ★ 필수 열이 없으면 예외를 던져야 한다."""
    bad = tmp_path / "bad.csv"
    bad.write_text("design,seq_len,cycles\nexact,128,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="필수 열 누락"):
        parse_sim_csv(bad)


def test_crosscheck_detects_mismatch():
    measured = parse_sim_csv(MOCK / "rtl_simulation_cycles.csv")
    # 예측이 실측과 정확히 같으면 통과해야 한다
    predicted = [
        {"design": m["design"], "seq_len": m["seq_len"], "top_k": m["top_k"],
         "margin": m["margin"], "total_cycles": m["cycles"], "words_bram": m["bram_reads"]}
        for m in measured
    ]
    res = compare(predicted, measured, tolerance=0.05)
    assert res["passed"] and res["n_mismatch"] == 0

    # 20% 어긋나면 잡아내야 한다
    for p in predicted:
        p["total_cycles"] = int(p["total_cycles"] * 1.2)
    res2 = compare(predicted, measured, tolerance=0.05)
    assert not res2["passed"]
    assert res2["n_mismatch"] > 0
    assert "점검할 것" in report(res2)


def test_crosscheck_reports_unmatched():
    measured = parse_sim_csv(MOCK / "rtl_simulation_cycles.csv")
    res = compare([], measured, tolerance=0.05)
    assert res["n_unmatched"] == len(measured)
    assert not res["passed"]
