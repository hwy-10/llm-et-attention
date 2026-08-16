"""EXP6 — ★ 손익분기 ★  배경지식 가이드 6.3-(4), 7.3

    비용 : 제어 논리 면적, 동작 주파수 저하, 파이프라인 빈 구간
    이득 : 연산 사이클 감소, 메모리 읽기 감소
    -> 어떤 조건에서 이득이 비용을 넘어서는가?

사이클만 보면 안 된다. Fmax 가 함께 떨어지므로

    실효 speedup = (cycles_기준 / cycles_제안) x (fmax_제안 / fmax_기준)

이 1.0 을 넘는 구간을 찾아야 한다. Vivado 실측이 아직 없으면
config/hardware.yaml 의 추정치로 돌고, fmax_derate 스윕으로 결론이
얼마나 견고한지 함께 본다.

★ 정직하게 짚을 것 ★
비트평면 순차 처리는 8사이클을 쓰므로, 기준 설계(①) 대비 사이클에서
구조적으로 불리하다. 제안의 근거는 사이클이 아니라
(a) DSP 미사용, (b) 메모리 읽기 감소 다. 이 표가 그 사실을 숨기지 않는다.
"""

from __future__ import annotations

import numpy as np

from src.decode_loop import run_decode
from utils.cost_model import ResourceReport, breakeven_curve, compare, control_overhead
from utils.io import save_records

from . import bram_from_config, build_workbench, load_config, spec_from_config

NAME = "exp6_breakeven"
_DESIGN_TO_RESOURCE = {
    "baseline": "baseline",
    "seq": "seq_no_et",
    "exact": "exact_et",
    "approx": "approx_et",
}


def run(cfg=None, verbose: bool = True) -> list[dict]:
    cfg = cfg or load_config()
    sw = cfg.get("sweeps.exp6", {}) or {}
    seq_lens = sw.get("seq_len", [128, 256, 512])
    top_ks = sw.get("top_k", [8])
    margins = sw.get("margin", [0.0, 0.1, 0.2])
    derates = sw.get("fmax_derate_sweep", [1.0, 0.95, 0.9, 0.85, 0.8, 0.75])

    sched, bram = spec_from_config(cfg), bram_from_config(cfg)
    res = {d: ResourceReport.from_config(cfg, key) for d, key in _DESIGN_TO_RESOURCE.items()}
    records: list[dict] = []

    for seq_len in seq_lens:
        wb = build_workbench(cfg, seq_len=int(seq_len))
        cycles: dict[tuple, float] = {}

        for design in ("baseline", "seq"):
            r = run_decode(wb, design=design, top_k=int(top_ks[0]), margin=0.0,
                           sched=sched, bram=bram, keep_trace=False)
            cycles[(design, 0.0, top_ks[0])] = r.summary["total_cycles"]

        for top_k in top_ks:
            for margin in margins:
                design = "exact" if margin == 0.0 else "approx"
                r = run_decode(wb, design=design, top_k=int(top_k), margin=float(margin),
                               sched=sched, bram=bram, keep_trace=False)
                cycles[(design, float(margin), int(top_k))] = r.summary["total_cycles"]
                base_key = ("baseline", 0.0, top_ks[0])
                seq_key = ("seq", 0.0, top_ks[0])

                for derate in derates:
                    # ① 대비 (전체 시스템 관점)
                    c1 = compare(res["baseline"], res[design],
                                 cycles[base_key], cycles[(design, float(margin), int(top_k))],
                                 fmax_derate=float(derate))
                    # ② 대비 (★ 종단의 순수 이득. 가이드 8.1절이 강조한 비교 ★)
                    c2 = compare(res["seq"], res[design],
                                 cycles[seq_key], cycles[(design, float(margin), int(top_k))],
                                 fmax_derate=float(derate))
                    rec = {
                        "seq_len": int(seq_len), "top_k": int(top_k),
                        "margin": float(margin), "design": design,
                        "fmax_derate": float(derate),
                    }
                    rec.update({f"vs1_{k}": v for k, v in c1.items()})
                    rec.update({f"vs2_{k}": v for k, v in c2.items()})
                    rec.update(control_overhead(res["seq"], res[design]))
                    records.append(rec)

    save_records(records, NAME, cfg)
    if verbose:
        _report(records, cfg)
    return records


def _report(records: list[dict], cfg) -> None:
    print("\n  실효 speedup (derate=1.0 기준):")
    print(f"    {'N':>5s} {'k':>3s} {'m':>5s} {'design':<8s} {'vs①':>8s} {'vs②':>8s} {'허용 Fmax비':>12s}")
    for r in records:
        if r["fmax_derate"] != 1.0:
            continue
        print(f"    {r['seq_len']:>5d} {r['top_k']:>3d} {r['margin']:>5.2f} {r['design']:<8s} "
              f"{r['vs1_effective_speedup']:>8.3f} {r['vs2_effective_speedup']:>8.3f} "
              f"{r['vs2_max_tolerable_fmax_ratio']:>12.3f}")

    wins2 = [r for r in records if r["vs2_effective_speedup"] > 1.0]
    print(f"\n  ② 대비 이득이 남는 조합: {len(wins2)} / {len(records)}")
    wins1 = [r for r in records if r["vs1_effective_speedup"] > 1.0]
    print(f"  ① 대비 이득이 남는 조합: {len(wins1)} / {len(records)}")
    if not wins1:
        print("    -> 비트평면 순차 처리는 사이클에서 구조적으로 불리하다.")
        print("       제안의 근거는 사이클이 아니라 (a) DSP 미사용 (b) 메모리 읽기 감소다.")
    warns = cfg.provenance_warnings()
    if warns:
        print(f"\n  ⚠ config/hardware.yaml 에 추정치 {len(warns)}개 — Vivado 실측으로 교체 필요:")
        for w in warns[:8]:
            print(f"      {w}")


def breakeven_grid(cycle_ratios=None, fmax_ratios=None):
    """그림용 등고선 격자. utils.visualization.fig_breakeven 에 넣는다."""
    cycle_ratios = cycle_ratios if cycle_ratios is not None else np.linspace(0.4, 1.2, 41)
    fmax_ratios = fmax_ratios if fmax_ratios is not None else np.linspace(0.6, 1.1, 41)
    grid = breakeven_curve(1.0, cycle_ratios, fmax_ratios)
    return grid, cycle_ratios, fmax_ratios


def main() -> int:
    from utils.io import enable_utf8_stdout

    enable_utf8_stdout()
    print(f"=== {NAME} ===")
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
