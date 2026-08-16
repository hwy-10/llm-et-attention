"""Vivado 보고서 / RTL 시뮬레이션 로그 파싱.

★ 권장 ★
Vivado 보고서 포맷은 버전마다 달라 파서가 잘 깨진다.
가능하면 RTL 팀이 **요약 JSON 을 직접 뱉게** 하고 (rtl_data/schema.md 참조),
이 파서는 폴백으로 쓰는 것이 안전하다. load_report() 가 둘을 자동 판별한다.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 자원 사용량 (report_utilization)
# ---------------------------------------------------------------------------
_UTIL_KEYS = {
    "lut": (r"Slice LUTs", r"CLB LUTs", r"LUT as Logic"),
    "ff": (r"Slice Registers", r"CLB Registers", r"Register as Flip Flop"),
    "dsp": (r"DSPs", r"DSP48E1", r"DSP48E2"),
    "bram36": (r"Block RAM Tile", r"RAMB36/FIFO"),
}


def _table_value(text: str, patterns: tuple[str, ...]) -> float | None:
    """'| Slice LUTs | 8200 | 0 | ...' 형태의 행에서 Used 값을 뽑는다."""
    for pat in patterns:
        m = re.search(
            rf"\|\s*{pat}[^|]*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|", text, re.IGNORECASE
        )
        if m:
            return float(m.group(1))
    return None


def parse_utilization(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    out: dict[str, Any] = {}
    for key, pats in _UTIL_KEYS.items():
        v = _table_value(text, pats)
        if v is not None:
            out[key] = int(v) if key != "bram36" else float(v)
    return out


# ---------------------------------------------------------------------------
# 타이밍 (report_timing_summary)
# ---------------------------------------------------------------------------
_NUM = re.compile(r"^\s*(-?[0-9]+(?:\.[0-9]+)?)\s*$")


def _column_value(text: str, header_key: str, lookahead: int = 6) -> float | None:
    """'| WNS (ns) | TNS (ns) | ... |' 같은 표에서 해당 열의 첫 데이터 값을 뽑는다.

    열 위치를 헤더에서 찾아 같은 인덱스의 셀을 읽으므로, 열 순서가 바뀌거나
    구분선이 끼어도 견딘다. (Vivado 포맷이 버전마다 다른 것에 대한 방어)
    """
    lines = text.splitlines()
    key = header_key.lower()
    for i, line in enumerate(lines):
        if "|" not in line or key not in line.lower():
            continue
        cells = [c.strip() for c in line.split("|")]
        idx = next((j for j, c in enumerate(cells) if key in c.lower()), None)
        if idx is None:
            continue
        for nxt in lines[i + 1 : i + 1 + lookahead]:
            if "|" not in nxt or set(nxt.strip()) <= set("|-+ "):
                continue
            cs = [c.strip() for c in nxt.split("|")]
            if idx < len(cs):
                m = _NUM.match(cs[idx])
                if m:
                    return float(m.group(1))
    return None


def parse_timing(path: str | Path, clock_period_ns: float | None = None) -> dict:
    """WNS 와 클럭 주기로부터 Fmax 를 계산한다.

        Fmax = 1 / (T_clk - WNS)
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    out: dict[str, Any] = {}

    wns = _column_value(text, "WNS")
    if wns is None:
        m = re.search(r"\bWNS\s*\(ns\)\s*[:=]\s*(-?[0-9]+\.[0-9]+)", text, re.IGNORECASE)
        wns = float(m.group(1)) if m else None
    if wns is not None:
        out["wns_ns"] = wns

    if clock_period_ns is None:
        clock_period_ns = _column_value(text, "Period")
    if clock_period_ns is None:
        mc = re.search(r"create_clock[^\n]*-period\s+([0-9]+\.?[0-9]*)", text)
        if mc:
            clock_period_ns = float(mc.group(1))

    if clock_period_ns and "wns_ns" in out:
        eff = clock_period_ns - out["wns_ns"]
        out["clock_period_ns"] = clock_period_ns
        out["fmax_mhz"] = 1000.0 / eff if eff > 0 else float("inf")
    return out


# ---------------------------------------------------------------------------
# 전력 (report_power)
# ---------------------------------------------------------------------------
def parse_power(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    out: dict[str, Any] = {}
    m = re.search(r"\|\s*Dynamic\s*\(W\)\s*\|\s*([0-9]+\.[0-9]+)", text)
    if m:
        out["dynamic_power_mw"] = float(m.group(1)) * 1000.0
    m = re.search(r"\|\s*Device Static\s*\(W\)\s*\|\s*([0-9]+\.[0-9]+)", text)
    if m:
        out["static_power_mw"] = float(m.group(1)) * 1000.0
    return out


# ---------------------------------------------------------------------------
# 통합 로더
# ---------------------------------------------------------------------------
def load_report(path: str | Path, clock_period_ns: float | None = None) -> dict:
    """.json 이면 그대로, .rpt 면 세 파서를 모두 시도해 합친다.

    RTL 팀이 요약 JSON 을 주는 경우가 가장 안전하다 (rtl_data/schema.md).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    out: dict[str, Any] = {"source_file": path.name}
    for fn in (parse_utilization, parse_power):
        try:
            out.update(fn(path))
        except Exception:
            pass
    try:
        out.update(parse_timing(path, clock_period_ns))
    except Exception:
        pass
    return out


def load_all_reports(rtl_dir: str | Path, clock_period_ns: float | None = None) -> dict:
    """rtl_data/ 아래의 모든 보고서를 설계 이름별로 모은다.

    파일명 규약: <design>_synth.rpt / <design>_timing.rpt / <design>_power.rpt
                 또는 <design>.json
    """
    rtl_dir = Path(rtl_dir)
    merged: dict[str, dict] = {}
    for p in sorted(rtl_dir.glob("*")):
        if p.suffix.lower() not in (".rpt", ".json"):
            continue
        design = re.sub(r"_(synth|timing|power|report)$", "", p.stem)
        merged.setdefault(design, {}).update(load_report(p, clock_period_ns))
    return merged


# ---------------------------------------------------------------------------
# RTL 시뮬레이션 사이클/읽기 로그
# ---------------------------------------------------------------------------
REQUIRED_SIM_COLUMNS = ("design", "seq_len", "top_k", "margin", "cycles", "bram_reads")


def parse_sim_csv(path: str | Path) -> list[dict]:
    """테스트벤치가 뱉은 단계별 사이클/읽기 CSV.

    필수 열은 REQUIRED_SIM_COLUMNS. 없으면 예외를 던져 조용한 실패를 막는다.
    """
    path = Path(path)
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_SIM_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{path.name}: 필수 열 누락 {missing}. rtl_data/schema.md 를 따르세요. "
                f"(발견된 열: {reader.fieldnames})"
            )
        for r in reader:
            rows.append({k: _num(v) for k, v in r.items()})
    return rows


def _num(v: str) -> Any:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v
