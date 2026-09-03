"""논문 실험 오케스트레이터.

    python run_paper_experiments.py                 # 전부 실행 + 그림
    python run_paper_experiments.py --only exp1 exp2
    python run_paper_experiments.py --skip exp5 exp6
    python run_paper_experiments.py --figures-only  # 실험 재실행 없이 그림만
    python run_paper_experiments.py --crosscheck    # RTL 실측과 대조
    python run_paper_experiments.py --quick         # 짧은 시퀀스로 빠르게 점검

★ 원칙 ★
실험은 outputs/raw/ 에 원본을 쓰고, 그림은 outputs/raw/ 만 읽는다.
따라서 --figures-only 로 실험 없이 figure 를 언제든 다시 뽑을 수 있다.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config                              # noqa: E402
from utils.io import ensure_dirs, enable_utf8_stdout, load_records, save_records  # noqa: E402

EXPERIMENTS = {
    "exp1": ("experiments.exp1_termination_profile", "종단 프로파일 (가장 먼저 실행)"),
    "exp2": ("experiments.exp2_margin_sweep", "margin 스윕 — 그림 8.1"),
    "exp3": ("experiments.exp3_theta_policy", "θ 확정 시점 (6.3-3)"),
    "exp4": ("experiments.exp4_schedule_policy", "스케줄 정책 + BRAM 워드폭 (6.3-2)"),
    "exp5": ("experiments.exp5_seqlen_topk", "문맥 길이 / 상위 k 스캔"),
    "exp6": ("experiments.exp6_breakeven", "손익분기 (6.3-4)"),
    "exp7": ("experiments.exp7_memory_bottleneck", "★ 병목 위치와 종단의 가치 (팀2)"),
    "exp8": ("experiments.exp8_real_perplexity", "★ 실측 perplexity — 보간 아님 (팀2)"),
    "exp9": ("experiments.exp9_margin_coverage", "★ margin 커버리지 — 전 헤드 x 여러 텍스트 (팀2)"),
}


def _import(mod: str):
    import importlib

    return importlib.import_module(mod)


# ---------------------------------------------------------------------------
def run_experiments(names: list[str], cfg, quick: bool = False) -> dict[str, int]:
    from experiments import build_workbench

    status: dict[str, int] = {}
    shared_wb = None
    for key in names:
        mod_name, desc = EXPERIMENTS[key]
        print(f"\n{'=' * 68}\n[{key}] {desc}\n{'=' * 68}", flush=True)
        t0 = time.perf_counter()
        try:
            mod = _import(mod_name)
            if key in ("exp5", "exp6", "exp7", "exp8", "exp9"):
                mod.run(cfg)                       # 자체적으로 seq_len 을 바꾼다
            else:
                if shared_wb is None:
                    seq = 128 if quick else None
                    shared_wb = build_workbench(cfg, seq_len=seq)
                    print(f"  (워크벤치 준비 {shared_wb.build_seconds:.2f}s, "
                          f"부분내적 {shared_wb.partials.shape})", flush=True)
                mod.run(cfg, wb=shared_wb)
            status[key] = 0
            print(f"\n  완료 ({time.perf_counter() - t0:.1f}s)", flush=True)
        except Exception:
            status[key] = 1
            print(f"\n  실패 ({time.perf_counter() - t0:.1f}s)", file=sys.stderr)
            traceback.print_exc()
    return status


# ---------------------------------------------------------------------------
def make_figures(cfg) -> int:
    """outputs/raw/ 만 읽어 그림을 만든다."""
    try:
        from utils import visualization as viz
    except Exception as exc:
        print(f"[figures] {exc}", file=sys.stderr)
        return 1

    made = 0
    # 그림 8.1 — margin 트레이드오프
    try:
        recs = load_records("exp2_margin_sweep")
        default_k = cfg.get("sweeps.exp2.top_k", [16])[0]   # ★ K_TOP 확정값
        sub = [r for r in recs if r.get("top_k") == default_k
               and r.get("design") in ("exact", "approx")]
        if sub:
            viz.fig_tradeoff(sub, accuracy_key=f"top{default_k}_retention",
                             title=f"θ 여유값 대비 절감량과 정확도 (top-{default_k})")
            made += 1
    except FileNotFoundError as e:
        print(f"  건너뜀: {e}")

    # 종단 프로파일
    try:
        import numpy as np
        from utils.io import RAW_DIR

        traces = {}
        for p in sorted(RAW_DIR.glob("exp1_termination_profile_trace_*.npz")):
            key = p.stem.replace("exp1_termination_profile_trace_", "")
            d = np.load(p)
            traces[key.split("_")[0]] = {k: d[k] for k in d.files}
        if traces:
            viz.fig_termination_profile(traces, n_planes=cfg.n_planes)
            made += 1
    except Exception as e:
        print(f"  건너뜀 (종단 프로파일): {e}")

    # BRAM 워드폭 실현률
    try:
        recs = load_records("exp4_schedule_policy")
        sub = [r for r in recs if r.get("margin") == 0.0]
        if sub:
            viz.fig_read_realization(sub)
            made += 1
    except FileNotFoundError as e:
        print(f"  건너뜀: {e}")

    # 손익분기 등고선
    try:
        from experiments.exp6_breakeven import breakeven_grid

        grid, cr, fr = breakeven_grid()
        viz.fig_breakeven(grid, cr, fr)
        made += 1
    except Exception as e:
        print(f"  건너뜀 (손익분기): {e}")

    # 설계 4종 비교 막대
    try:
        recs = load_records("exp2_margin_sweep")
        pick = []
        for d in ("baseline", "seq", "exact", "approx"):
            cand = [r for r in recs if r.get("design") == d]
            if cand:
                pick.append(cand[0])
        if pick:
            viz.fig_design_comparison(pick, metric="total_cycles", ylabel="총 사이클")
            made += 1
    except FileNotFoundError as e:
        print(f"  건너뜀: {e}")

    print(f"\n  그림 {made}개 생성 -> outputs/figures/")
    return 0 if made else 1


# ---------------------------------------------------------------------------
def make_tables(cfg) -> int:
    """가이드 8.4절의 '네 설계 비교표'."""
    from utils.io import to_latex_table

    try:
        recs = load_records("exp2_margin_sweep")
    except FileNotFoundError as e:
        print(f"  건너뜀: {e}")
        return 1

    pick = []
    for d in ("baseline", "seq", "exact", "approx"):
        cand = [r for r in recs if r.get("design") == d]
        if cand:
            pick.append(max(cand, key=lambda r: r.get("read_saving_bram", 0.0)))

    cols = [
        ("design", "설계"),
        ("total_cycles", "사이클"),
        ("words_bram", "BRAM 읽기"),
        ("read_saving_bram", "읽기 절감(실현)"),
        ("read_saving_ideal", "읽기 절감(이론)"),
        ("mean_term_plane", "평균 종단 평면"),
        ("top8_retention", "top-8 보존율"),
    ]
    to_latex_table(pick, cols, "table_design_comparison",
                   caption="네 설계 비교 (동일 조건)", label="tab:design")
    print("  표 생성 -> outputs/tables/table_design_comparison.{tex,csv}")
    return 0


# ---------------------------------------------------------------------------
def predict_for(cfg, keys: list[dict]) -> list[dict]:
    """실측 레코드의 키 조합에 정확히 대응하는 예측을 만든다.

    exp5 의 스윕 범위에 의존하지 않으므로 키가 어긋날 일이 없다.
    seq_len 별로 워크벤치를 한 번씩만 만든다.
    """
    from experiments import build_workbench
    from src.decode_loop import run_decode
    from src.schedule import bram_from_config, spec_from_config

    sched = spec_from_config(cfg)
    by_len: dict[int, object] = {}
    out: list[dict] = []

    for k in keys:
        n = int(k["seq_len"])
        if n not in by_len:
            by_len[n] = build_workbench(cfg, seq_len=n)
        bram = bram_from_config(
            cfg, **({"word_tokens": int(k["word_tokens"])} if k.get("word_tokens") else {})
        )
        r = run_decode(
            by_len[n], design=str(k["design"]), top_k=int(k["top_k"]),
            margin=float(k["margin"]), sched=sched, bram=bram, keep_trace=False,
        )
        out.append({
            "design": k["design"], "seq_len": n, "top_k": int(k["top_k"]),
            "margin": float(k["margin"]),
            "total_cycles": r.summary["total_cycles"],
            "words_bram": r.summary["words_bram"],
            "mean_term_plane": r.summary["mean_term_plane"],
        })
    return out


def run_crosscheck(cfg, tolerance: float = 0.05) -> int:
    """SW 예측 vs RTL 실측."""
    from utils import crosscheck
    from utils.hw_parser import parse_sim_csv
    from src.config import PROJECT_ROOT

    sim = PROJECT_ROOT / "rtl_data" / "rtl_simulation_cycles.csv"
    if not sim.exists():
        sim = PROJECT_ROOT / "rtl_data" / "mock" / "rtl_simulation_cycles.csv"
        print(f"  실측 파일이 없어 mock 을 사용합니다: {sim}")
        print("  (mock 은 소프트웨어 모델에서 생성한 것이므로 통과하는 것이 정상입니다.\n"
              "   실제 RTL 실측을 rtl_data/ 최상위에 놓으면 진짜 대조가 됩니다.)")
    if not sim.exists():
        print("  rtl_data/ 에 시뮬레이션 CSV 가 없습니다. rtl_data/schema.md 참조.")
        return 1

    measured = parse_sim_csv(sim)
    predicted = predict_for(cfg, measured)
    res = crosscheck.compare(predicted, measured, tolerance=tolerance)
    print(crosscheck.report(res))
    return 0 if res["passed"] else 1


def generate_mock(cfg) -> int:
    """소프트웨어 모델에서 mock 실측 CSV 를 만든다 (RTL 없이 배관 점검용)."""
    import csv as _csv

    from src.config import PROJECT_ROOT

    # ★ word_tokens 를 여기 하드코딩하면 config 를 바꿔도 mock 이 안 따라온다.
    #   2026-08-28 WORD_TOKENS=1 확정 후 이 값이 32 로 남아 mock 만 옛 설정으로
    #   생성되던 문제가 있었다. 설정에서 읽어 온다.
    wt = int((cfg.get("hardware.memory", {}) or {}).get("word_tokens", 1))
    keys = [
        {"design": d, "seq_len": n, "top_k": 8,
         "margin": 0.1 if d == "approx" else 0.0, "word_tokens": wt}
        for n in (128, 256, 512)
        for d in ("baseline", "seq", "exact", "approx")
    ]
    preds = predict_for(cfg, keys)
    out = PROJECT_ROOT / "rtl_data" / "mock" / "rtl_simulation_cycles.csv"
    cols = ["design", "seq_len", "top_k", "margin", "cycles", "bram_reads",
            "mean_term_plane", "word_tokens"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(cols)
        for k, p in zip(keys, preds):
            w.writerow([p["design"], p["seq_len"], p["top_k"], p["margin"],
                        p["total_cycles"], p["words_bram"],
                        f"{p['mean_term_plane']:.2f}", k["word_tokens"]])
    print(f"  mock 생성 -> {out}  ({len(preds)} 행)")
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    enable_utf8_stdout()
    ap = argparse.ArgumentParser(description="논문 실험 전체 실행")
    ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=None)
    ap.add_argument("--skip", nargs="+", choices=list(EXPERIMENTS), default=[])
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--crosscheck", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.05, help="crosscheck 허용 오차")
    ap.add_argument("--generate-mock", action="store_true",
                    help="소프트웨어 모델에서 mock 실측 CSV 를 다시 생성")
    ap.add_argument("--quick", action="store_true", help="짧은 시퀀스로 빠르게 점검")
    args = ap.parse_args(argv)

    ensure_dirs()
    cfg = load_config()
    print(f"설정 해시 {cfg.hash()} | head_dim={cfg.head_dim} planes={cfg.n_planes} "
          f"seq_len={cfg.seq_len}")
    warns = cfg.provenance_warnings()
    if warns:
        print(f"⚠ 하드웨어 추정치 {len(warns)}개 (Vivado 실측으로 교체 필요)")

    if args.generate_mock:
        return generate_mock(cfg)
    if args.crosscheck:
        return run_crosscheck(cfg, tolerance=args.tolerance)

    rc = 0
    if not args.figures_only:
        names = [k for k in (args.only or list(EXPERIMENTS)) if k not in args.skip]
        status = run_experiments(names, cfg, quick=args.quick)
        rc = max(status.values()) if status else 0
        print(f"\n{'=' * 68}")
        for k, v in status.items():
            print(f"  {k}: {'OK' if v == 0 else '실패'}")

    if not args.no_figures:
        print(f"\n{'=' * 68}\n[figures] outputs/raw -> outputs/figures\n{'=' * 68}")
        make_figures(cfg)
        make_tables(cfg)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
