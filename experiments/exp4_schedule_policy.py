"""EXP4 — 종단 불규칙성 처리 + ★ BRAM 워드폭 함정 ★
배경지식 가이드 6.3-(2), 5.7

이 실험이 이 프로젝트에서 가장 반박당하기 쉬운 지점을 방어한다.

가이드 5.7절은 "메모리 읽기 절감이 더 중요하다"고 했다. 맞다.
그런데 BRAM 은 토큰을 하나씩 읽지 않는다. 한 워드에 word_tokens 개 토큰의
같은 평면 비트가 묶여 있으므로,

    워드 안에 살아있는 토큰이 하나라도 있으면 워드 전체를 읽어야 한다.

따라서 종단된 토큰이 흩어져 있으면 이론 절감이 실현되지 않는다.
work-compaction 이 필요한 진짜 이유가 이것이며, 이 실험이 그 격차를 정량화한다.

  x축   : word_tokens (1 = 이상적, 32/64 = 현실적)
  계열  : 스케줄 정책 (batch / compaction / two_phase)
  y축   : 실현되는 읽기 절감률
"""

from __future__ import annotations

from src.decode_loop import run_decode
from utils.io import save_records

from . import bram_from_config, build_workbench, load_config, spec_from_config

NAME = "exp4_schedule_policy"


def run(cfg=None, wb=None, verbose: bool = True) -> list[dict]:
    cfg = cfg or load_config()
    wb = wb or build_workbench(cfg)
    sw = cfg.get("sweeps.exp4", {}) or {}
    policies = sw.get("schedule_policy", ["batch", "compaction", "two_phase"])
    batch_sizes = sw.get("batch_size", [8, 16, 32, 64])
    splits = sw.get("two_phase_split", [2, 3, 4])
    word_tokens = sw.get("word_tokens", [1, 8, 16, 32, 64])
    top_ks = sw.get("top_k", [8])
    margins = sw.get("margin", [0.0, 0.1])

    records: list[dict] = []
    for pol in policies:
        variants = (
            [("batch_size", b) for b in batch_sizes] if pol == "batch"
            else [("two_phase_split", s) for s in splits] if pol == "two_phase"
            else [("", 0)]
        )
        for vname, vval in variants:
            sched = spec_from_config(cfg, **({vname: int(vval)} if vname else {}))
            for wt in word_tokens:
                bram = bram_from_config(cfg, word_tokens=int(wt))
                for top_k in top_ks:
                    for margin in margins:
                        design = "exact" if margin == 0.0 else "approx"
                        r = run_decode(
                            wb, design=design, top_k=int(top_k), margin=float(margin),
                            schedule_policy=pol, sched=sched, bram=bram, keep_trace=False,
                        )
                        rec = r.record()
                        rec["margin"] = float(margin)
                        rec["batch_size"] = sched.batch_size
                        rec["two_phase_split"] = sched.two_phase_split
                        rec["variant"] = f"{vname}={vval}" if vname else ""
                        records.append(rec)

    save_records(records, NAME, cfg)
    if verbose:
        _report_realization(records)
    return records


def _report_realization(records: list[dict]) -> None:
    print("\n  워드폭에 따른 읽기 절감 실현률 (margin=0, 정확 모드):")
    print(f"    {'wt':>4s} {'policy':<12s} {'이론':>7s} {'실현':>7s} {'실현률':>8s} {'사이클':>10s}")
    rows = [r for r in records if r.get("margin") == 0.0]
    for r in sorted(rows, key=lambda x: (x.get("word_tokens", 0), x.get("schedule_policy", ""))):
        ideal = r.get("read_saving_ideal", 0.0)
        real = r.get("read_saving_bram", 0.0)
        ratio = real / ideal if ideal > 0 else 1.0
        print(f"    {r.get('word_tokens'):>4d} {r.get('schedule_policy',''):<12s} "
              f"{ideal * 100:>6.1f}% {real * 100:>6.1f}% {ratio * 100:>7.0f}% "
              f"{r.get('total_cycles', 0):>10,d}")

    worst = min(
        (r for r in rows if r.get("read_saving_ideal", 0) > 0),
        key=lambda r: r["read_saving_bram"] / r["read_saving_ideal"],
        default=None,
    )
    if worst:
        ratio = worst["read_saving_bram"] / worst["read_saving_ideal"]
        print(
            f"\n    ★ 최악 조합: word_tokens={worst['word_tokens']}, "
            f"policy={worst['schedule_policy']} -> 이론 절감의 {ratio * 100:.0f}% 만 실현"
        )


def main() -> int:
    from utils.io import enable_utf8_stdout

    enable_utf8_stdout()
    print(f"=== {NAME} ===")
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
