"""자원 · 주파수를 감안한 손익분기 — 배경지식 가이드 6.3-(4)절

조기 종단은 공짜가 아니다.

    비용 : 제어 논리 LUT, 동작 주파수 저하, 파이프라인 빈 구간
    이득 : 연산 사이클 감소, 메모리 읽기 감소

★ 사이클만 보면 안 된다 ★
Fmax 가 20% 떨어지고 사이클이 25% 줄면 실효 이득은 거의 0이다.

    실효 speedup = (cycles_기준 / cycles_제안) x (fmax_제안 / fmax_기준)

이 계산이 이 프로젝트의 결론에 해당한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class ResourceReport:
    """한 설계의 자원·주파수·전력. Vivado 보고서 또는 config 에서 온다."""

    name: str = ""
    lut: int = 0
    ff: int = 0
    dsp: int = 0
    bram36: float = 0.0
    fmax_mhz: float = 0.0
    dynamic_power_mw: float = 0.0
    source: str = "estimate"

    @classmethod
    def from_config(cls, cfg, design_key: str) -> "ResourceReport":
        r = (cfg.get(f"hardware.resources.{design_key}", {}) or {})
        return cls(
            name=design_key,
            lut=int(r.get("lut", 0)),
            ff=int(r.get("ff", 0)),
            dsp=int(r.get("dsp", 0)),
            bram36=float(r.get("bram36", 0.0)),
            fmax_mhz=float(r.get("fmax_mhz", 0.0)),
            dynamic_power_mw=float(r.get("dynamic_power_mw", 0.0)),
            source=str(r.get("source", "estimate")),
        )

    def derate(self, factor: float) -> "ResourceReport":
        """Fmax 를 factor 배로 낮춘 가상 보고서 (민감도 분석용)."""
        return replace(self, fmax_mhz=self.fmax_mhz * factor, source=f"{self.source}+derate")


# ---------------------------------------------------------------------------
def effective_speedup(
    cycles_ref: float, fmax_ref: float, cycles_new: float, fmax_new: float
) -> float:
    """실효 속도 배율. 1.0 보다 커야 이득이다."""
    if cycles_new <= 0 or fmax_ref <= 0:
        return 0.0
    return (cycles_ref / cycles_new) * (fmax_new / fmax_ref)


def latency_us(cycles: float, fmax_mhz: float) -> float:
    return cycles / fmax_mhz if fmax_mhz > 0 else float("inf")


def energy_proxy_uj(cycles: float, fmax_mhz: float, dynamic_power_mw: float) -> float:
    """동적 에너지 근사 = 전력 x 시간. Vivado 전력 추정치를 그대로 쓴다."""
    return dynamic_power_mw * latency_us(cycles, fmax_mhz) * 1e-3


def max_tolerable_derate(cycles_ref: float, cycles_new: float) -> float:
    """실효 speedup 1.0 을 유지하기 위해 허용되는 최소 Fmax 비율.

    예: 사이클이 25% 줄면(0.75배) Fmax 는 0.75 배까지만 떨어져도 본전.
    """
    if cycles_ref <= 0:
        return 1.0
    return cycles_new / cycles_ref


def compare(
    ref: ResourceReport,
    new: ResourceReport,
    cycles_ref: float,
    cycles_new: float,
    fmax_derate: float = 1.0,
) -> dict:
    """두 설계를 사이클 + 자원 + 주파수까지 묶어 비교한다."""
    new_d = new.derate(fmax_derate) if fmax_derate != 1.0 else new
    sp = effective_speedup(cycles_ref, ref.fmax_mhz, cycles_new, new_d.fmax_mhz)
    e_ref = energy_proxy_uj(cycles_ref, ref.fmax_mhz, ref.dynamic_power_mw)
    e_new = energy_proxy_uj(cycles_new, new_d.fmax_mhz, new_d.dynamic_power_mw)
    return {
        "ref_design": ref.name,
        "new_design": new.name,
        "cycles_ref": cycles_ref,
        "cycles_new": cycles_new,
        "cycle_ratio": cycles_new / cycles_ref if cycles_ref else 0.0,
        "fmax_ref_mhz": ref.fmax_mhz,
        "fmax_new_mhz": new_d.fmax_mhz,
        "fmax_ratio": new_d.fmax_mhz / ref.fmax_mhz if ref.fmax_mhz else 0.0,
        "fmax_derate_applied": fmax_derate,
        "effective_speedup": sp,
        "is_win": sp > 1.0,
        "max_tolerable_fmax_ratio": max_tolerable_derate(cycles_ref, cycles_new),
        "latency_ref_us": latency_us(cycles_ref, ref.fmax_mhz),
        "latency_new_us": latency_us(cycles_new, new_d.fmax_mhz),
        "energy_ref_uj": e_ref,
        "energy_new_uj": e_new,
        "energy_ratio": e_new / e_ref if e_ref else 0.0,
        # 자원 교환 (가이드 6.3-(1)): DSP -> LUT
        "d_lut": new.lut - ref.lut,
        "d_ff": new.ff - ref.ff,
        "d_dsp": new.dsp - ref.dsp,
        "d_bram36": new.bram36 - ref.bram36,
        "lut_ratio": new.lut / ref.lut if ref.lut else 0.0,
        "dsp_saved": ref.dsp - new.dsp,
    }


def control_overhead(seq_no_et: ResourceReport, with_et: ResourceReport) -> dict:
    """조기 종단 제어 회로만의 오버헤드.

    BitStopper[4] 가 6.9% 로 보고한 항목과 같은 기준으로 비교할 수 있다.
    """
    lut_ov = (with_et.lut - seq_no_et.lut) / seq_no_et.lut if seq_no_et.lut else 0.0
    pw_ov = (
        (with_et.dynamic_power_mw - seq_no_et.dynamic_power_mw) / seq_no_et.dynamic_power_mw
        if seq_no_et.dynamic_power_mw
        else 0.0
    )
    fm_drop = (
        1.0 - with_et.fmax_mhz / seq_no_et.fmax_mhz if seq_no_et.fmax_mhz else 0.0
    )
    return {
        "control_lut_overhead": lut_ov,
        "control_ff_overhead": (with_et.ff - seq_no_et.ff) / seq_no_et.ff if seq_no_et.ff else 0.0,
        "control_power_overhead": pw_ov,
        "fmax_drop": fm_drop,
    }


def breakeven_curve(
    cycles_ref: float,
    cycles_new_list,
    fmax_ratio_list,
) -> np.ndarray:
    """(사이클비, Fmax비) 격자에서의 실효 speedup 표.

    반환 shape (len(cycles_new_list), len(fmax_ratio_list)).
    값이 1.0 인 등고선이 손익분기선이다.
    """
    cn = np.asarray(cycles_new_list, dtype=np.float64)[:, None]
    fr = np.asarray(fmax_ratio_list, dtype=np.float64)[None, :]
    return (cycles_ref / cn) * fr
