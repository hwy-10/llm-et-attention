"""EXP3 — θ 를 언제 확정할 것인가.  배경지식 가이드 6.3-(3)

    이르게 확정 -> θ 가 낮음 -> 종단 적음
    늦게 확정   -> θ 가 높음 -> 종단 많으나 이미 읽은 양이 많음

정책 네 가지를 같은 조건에서 비교한다.

    every_plane  : 매 평면 갱신 (하드웨어 비용 최대, 종단 최대)
    once_at_m    : 평면 m 개 시점에 한 번 확정하고 고정
    prev_step    : 직전 디코드 스텝의 k번째 점수를 재사용
                   -> 평면 0부터 종단 가능. 회로가 가장 단순하다.
                   ★ 디코드 루프를 돌아야만 평가할 수 있는 정책 ★
    oracle_fixed : 이번 스텝의 참 k번째 점수 (구현 불가, 상한 참조선)

prev_step 이 oracle 에 가깝게 나오면 "부분 정렬 회로 없이도 된다"는
설계 단순화 근거가 된다. 반대로 크게 밀리면 정렬 회로가 정당화된다.
"""

from __future__ import annotations

from src.decode_loop import run_decode
from utils.io import save_records

from . import bram_from_config, build_workbench, load_config, spec_from_config

NAME = "exp3_theta_policy"


def run(cfg=None, wb=None, verbose: bool = True) -> list[dict]:
    cfg = cfg or load_config()
    wb = wb or build_workbench(cfg)
    sw = cfg.get("sweeps.exp3", {}) or {}
    policies = sw.get("theta_policy", ["every_plane", "once_at_m", "prev_step", "oracle_fixed"])
    once_ms = sw.get("once_at_m", [1, 2, 3, 4])
    top_ks = sw.get("top_k", [8])
    margins = sw.get("margin", [0.0, 0.1])

    sched, bram = spec_from_config(cfg), bram_from_config(cfg)
    records: list[dict] = []

    for pol in policies:
        ms = once_ms if pol == "once_at_m" else [0]
        for m0 in ms:
            for top_k in top_ks:
                for margin in margins:
                    design = "exact" if margin == 0.0 else "approx"
                    r = run_decode(
                        wb, design=design, top_k=int(top_k), margin=float(margin),
                        theta_policy=pol, once_at_m=int(m0),
                        sched=sched, bram=bram, keep_trace=False,
                    )
                    rec = r.record()
                    rec["margin"] = float(margin)
                    records.append(rec)
                    if verbose:
                        s = r.summary
                        tag = f"{pol}" + (f"@{m0}" if pol == "once_at_m" else "")
                        print(
                            f"  {tag:<16s} k={top_k:<3d} m={margin:<4.2f} "
                            f"term={s['mean_term_plane']:.2f} "
                            f"read_bram={s['read_saving_bram']:.3f} "
                            f"top{top_k}={s.get(f'top{top_k}_retention', float('nan')):.4f}",
                            flush=True,
                        )

    save_records(records, NAME, cfg)
    if verbose:
        _compare_to_oracle(records)
    return records


def _compare_to_oracle(records: list[dict]) -> None:
    oracle = [r for r in records if r.get("theta_policy") == "oracle_fixed"]
    if not oracle:
        return
    best = max(oracle, key=lambda r: r.get("read_saving_bram", 0.0))
    ref = best.get("read_saving_bram", 0.0)
    print(f"\n  oracle 대비 실현 읽기 절감 (oracle = {ref * 100:.1f}%):")
    seen = set()
    for r in records:
        pol = r.get("theta_policy")
        tag = f"{pol}" + (f"@{r.get('once_at_m')}" if pol == "once_at_m" else "")
        if tag in seen:
            continue
        seen.add(tag)
        v = r.get("read_saving_bram", 0.0)
        print(f"    {tag:<16s} {v * 100:>5.1f}%  ({v / ref * 100:.0f}% of oracle)"
              if ref else f"    {tag:<16s} {v * 100:>5.1f}%")


def main() -> int:
    from utils.io import enable_utf8_stdout

    enable_utf8_stdout()
    print(f"=== {NAME} ===")
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
