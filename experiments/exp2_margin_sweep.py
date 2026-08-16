"""EXP2 — ★ 최종 산출물 ★  배경지식 가이드 8.4 / 그림 8.1

θ 여유값(margin)을 키워 가며 두 곡선을 얻는다.

    가로축: margin
    세로축 A: 절감량 (사이클, 메모리 읽기)
    세로축 B: 정확도 손실 (상위 k 보존율)

두 곡선을 겹쳐 "허용 가능한 손실 범위 안에서 절감이 가장 큰 지점"을 찾는다.

★ 절감량은 반드시 두 개를 함께 보고한다 ★
  read_saving_ideal : 살아있는 토큰 비율 (이론)
  read_saving_bram  : 실제 BRAM 워드 읽기 (실현)
"""

from __future__ import annotations

from src.decode_loop import run_decode
from utils.io import save_records

from . import bram_from_config, build_workbench, load_config, spec_from_config

NAME = "exp2_margin_sweep"


def run(cfg=None, wb=None, verbose: bool = True) -> list[dict]:
    cfg = cfg or load_config()
    wb = wb or build_workbench(cfg)
    sw = cfg.get("sweeps.exp2", {}) or {}
    margins = sw.get("margin", [0.0, 0.05, 0.1, 0.2, 0.4])
    top_ks = sw.get("top_k", [4, 8, 16])
    policies = sw.get("theta_policy", ["every_plane"])

    sched, bram = spec_from_config(cfg), bram_from_config(cfg)
    records: list[dict] = []

    # 기준선 두 개 (margin 축과 무관하지만 표에 함께 실린다)
    for design in ("baseline", "seq"):
        r = run_decode(wb, design=design, top_k=int(top_ks[0]), margin=0.0,
                       sched=sched, bram=bram, keep_trace=False)
        records.append(r.record())

    for top_k in top_ks:
        for pol in policies:
            for margin in margins:
                design = "exact" if margin == 0.0 else "approx"
                r = run_decode(
                    wb, design=design, top_k=int(top_k), margin=float(margin),
                    theta_policy=pol, sched=sched, bram=bram, keep_trace=False,
                )
                rec = r.record()
                rec["margin"] = float(margin)   # exact 도 margin 축에 올려 곡선을 잇는다
                records.append(rec)
                if verbose:
                    s = r.summary
                    print(
                        f"  k={top_k:<3d} m={margin:<5.2f} {design:<7s} "
                        f"read_bram={s['read_saving_bram']:.3f} "
                        f"cyc_vs_seq={s['cycle_saving_vs_seq']:.3f} "
                        f"top{top_k}={s.get(f'top{top_k}_retention', float('nan')):.4f} "
                        f"kl={s['softmax_kl']:.3e}",
                        flush=True,
                    )

    save_records(records, NAME, cfg)
    if verbose:
        _report_knee(records, top_ks)
    return records


def _report_knee(records: list[dict], top_ks) -> None:
    """허용 손실 안에서 절감이 최대인 지점 (그림 8.1 의 결론)."""
    print("\n  허용 손실별 최적 margin:")
    for tol in (0.0, 0.01, 0.05):
        for top_k in top_ks:
            key = f"top{top_k}_retention"
            cand = [
                r for r in records
                if r.get("design") in ("exact", "approx")
                and r.get("top_k") == top_k
                and (1.0 - r.get(key, 1.0)) <= tol + 1e-12
            ]
            if not cand:
                continue
            best = max(cand, key=lambda r: r.get("read_saving_bram", 0.0))
            print(
                f"    손실≤{tol * 100:>4.1f}%p, k={top_k:<3d} -> margin={best['margin']:.2f}, "
                f"읽기절감(실현)={best['read_saving_bram'] * 100:.1f}%, "
                f"이론={best['read_saving_ideal'] * 100:.1f}%"
            )


def main() -> int:
    from utils.io import enable_utf8_stdout

    enable_utf8_stdout()
    print(f"=== {NAME} ===")
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
