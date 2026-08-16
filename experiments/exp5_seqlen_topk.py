"""EXP5 — 문맥 길이 N 과 상위 k 스캔.  배경지식 가이드 6.3-(4)

    "문장 길이 N, 상위 k의 크기, 종단 정책에 따라 답이 달라집니다."

디코드 루프를 seq_len 별로 다시 돌린다 (워크벤치를 새로 만든다).
길이가 길어질수록 종단이 잘 일어나는지가 핵심 관찰이다.
문맥이 길수록 유리하다면 이 기법의 적용 구간이 명확해진다.
"""

from __future__ import annotations

from src.decode_loop import run_decode
from utils.io import save_records

from . import bram_from_config, build_workbench, load_config, spec_from_config

NAME = "exp5_seqlen_topk"


def run(cfg=None, verbose: bool = True) -> list[dict]:
    cfg = cfg or load_config()
    sw = cfg.get("sweeps.exp5", {}) or {}
    seq_lens = sw.get("seq_len", [128, 256, 512])
    top_ks = sw.get("top_k", [1, 2, 4, 8, 16, 32, 64])
    margins = sw.get("margin", [0.0, 0.1])
    policies = sw.get("theta_policy", ["every_plane"])

    sched, bram = spec_from_config(cfg), bram_from_config(cfg)
    records: list[dict] = []

    for seq_len in seq_lens:
        wb = build_workbench(cfg, seq_len=int(seq_len))
        for top_k in top_ks:
            if top_k > seq_len:
                continue
            for margin in margins:
                for pol in policies:
                    design = "exact" if margin == 0.0 else "approx"
                    r = run_decode(
                        wb, design=design, top_k=int(top_k), margin=float(margin),
                        theta_policy=pol, sched=sched, bram=bram, keep_trace=False,
                        eval_top_k=(1, 4, 8, 16),
                    )
                    rec = r.record()
                    rec["margin"] = float(margin)
                    rec["seq_len_cfg"] = int(seq_len)
                    records.append(rec)
                    if verbose:
                        s = r.summary
                        print(
                            f"  N={seq_len:<5d} k={top_k:<3d} m={margin:<4.2f} "
                            f"term={s['mean_term_plane']:.2f} "
                            f"read_bram={s['read_saving_bram']:.3f} "
                            f"cyc_vs_seq={s['cycle_saving_vs_seq']:+.3f}",
                            flush=True,
                        )

    save_records(records, NAME, cfg)
    if verbose:
        _report_trend(records, seq_lens)
    return records


def _report_trend(records: list[dict], seq_lens) -> None:
    print("\n  문맥 길이별 최대 실현 읽기 절감 (정확 모드):")
    for n in seq_lens:
        sub = [r for r in records if r.get("seq_len_cfg") == n and r.get("margin") == 0.0]
        if not sub:
            continue
        best = max(sub, key=lambda r: r.get("read_saving_bram", 0.0))
        print(f"    N={n:<5d} k={best['top_k']:<3d} 읽기절감 {best['read_saving_bram'] * 100:>5.1f}% "
              f"(이론 {best['read_saving_ideal'] * 100:.1f}%), "
              f"사이클 vs ② {best['cycle_saving_vs_seq'] * 100:+.1f}%")


def main() -> int:
    from utils.io import enable_utf8_stdout

    enable_utf8_stdout()
    print(f"=== {NAME} ===")
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
