"""EXP1 — 종단 프로파일.  배경지식 가이드 8.3 "1단계 알고리즘 수준 검증"

핵심 질문: **평균적으로 몇 개의 평면에서 종단되는가?**

이 실험이 먼저인 이유: 종단이 거의 일어나지 않으면 (평균 종단 평면 ≈ 8)
이후의 RTL 구현이 전부 무의미하므로, 설계를 다시 검토해야 한다.

함께 확인하는 것
  * 문맥 길이가 자라면서 종단이 더 잘 일어나는가 (디코드 루프를 도는 이유)
  * 정확 모드가 정말 무손실인가 (top-k 보존율 == 1.0 이어야 한다)
"""

from __future__ import annotations

import numpy as np

from src.decode_loop import run_decode
from utils.io import save_records, save_trace
from utils.metrics import termination_profile

from . import bram_from_config, build_workbench, load_config, spec_from_config

NAME = "exp1_termination_profile"


def run(cfg=None, wb=None, verbose: bool = True) -> list[dict]:
    cfg = cfg or load_config()
    wb = wb or build_workbench(cfg)
    sw = cfg.get("sweeps.exp1", {}) or {}
    designs = sw.get("designs", ["baseline", "seq", "exact", "approx"])
    top_ks = sw.get("top_k", [8])
    margins = sw.get("margin", [0.0, 0.1])
    policies = sw.get("theta_policy", ["every_plane"])

    sched, bram = spec_from_config(cfg), bram_from_config(cfg)
    records: list[dict] = []
    traces: dict[str, dict] = {}

    for design in designs:
        ms = margins if design == "approx" else [0.0]
        for top_k in top_ks:
            for margin in ms:
                for pol in policies:
                    r = run_decode(
                        wb, design=design, top_k=int(top_k), margin=float(margin),
                        theta_policy=pol, sched=sched, bram=bram, keep_trace=True,
                    )
                    rec = r.record()
                    # 종단 시점 분포를 함께 붙인다
                    rec.update(termination_profile(
                        np.repeat(r.per_step["mean_term_plane"], 1), wb.n_planes
                    ))
                    records.append(rec)
                    key = f"{design}" + (f"_m{margin}" if design == "approx" else "")
                    if key not in traces:
                        traces[key] = r.per_step
                    if verbose:
                        s = r.summary
                        print(
                            f"  {design:<9s} k={top_k:<3d} m={margin:<4.2f} "
                            f"term={s['mean_term_plane']:.2f}  "
                            f"read_ideal={s['read_saving_ideal']:.3f} "
                            f"read_bram={s['read_saving_bram']:.3f}  "
                            f"top{top_k}={s.get(f'top{top_k}_retention', float('nan')):.4f}",
                            flush=True,
                        )

    save_records(records, NAME, cfg)
    for key, tr in traces.items():
        save_trace(f"{NAME}_trace_{key}", tr)

    if verbose:
        _survival_curve(records, wb.n_planes)
        _sanity(records)
    return records


def _survival_curve(records: list[dict], n_planes: int) -> None:
    """평면별 생존 곡선 — 종단이 몇 번째 평면부터 시작되는가.

    2단계 처리(가이드 6.3-(2))의 분할점 m0 를 정하는 직접적 근거다.
    """
    print("\n  평면별 생존 비율 (평면 0 = MSB):")
    for r in records:
        if r["design"] not in ("exact", "approx"):
            continue
        vals = [r.get(f"live_frac_p{t}", 1.0) for t in range(n_planes)]
        bar = "  ".join(f"{v * 100:>4.0f}" for v in vals)
        tag = f"{r['design']}/m={r['margin']:.2f}"
        print(f"    {tag:<16s} {bar}   (첫 종단 평면 = {r.get('first_terminating_plane')})")


def _sanity(records: list[dict]) -> None:
    """★ 이 검사에 걸리면 이후 실험을 진행하면 안 된다 ★"""
    msgs = []
    for r in records:
        if r["design"] == "exact":
            k = r["top_k"]
            ret = r.get(f"top{k}_retention")
            if ret is not None and ret < 1.0 - 1e-9:
                msgs.append(f"정확 모드가 무손실이 아님: top{k}_retention={ret:.6f}")
            if r["mean_term_plane"] >= r.get("n_planes", 8) - 0.05:
                msgs.append("정확 모드에서 종단이 거의 일어나지 않음 — 설계 재검토 필요")
    if msgs:
        print("\n  [경고]")
        for m in dict.fromkeys(msgs):
            print(f"    ! {m}")
    else:
        print("\n  [OK] 정확 모드 무손실 확인, 종단 발생 확인")


def main() -> int:
    from utils.io import enable_utf8_stdout

    enable_utf8_stdout()
    print(f"=== {NAME} ===")
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
