"""EXP7 — ★ 병목 위치가 종단의 가치를 정한다 ★  (2026-08-29 신설, 팀 2)

    total_cycles = max(compute, memory)          mem_overlap = true
                 = compute + memory              mem_overlap = false

조기 종단이 줄이는 것은 **메모리 읽기**다. 그런데 위 식에서 `max()` 가
연산을 집으면, 메모리를 아무리 아껴도 시간이 줄지 않는다.

    memory_cycles = ceil(words_bram / n_ports)

이므로 병목 위치는 `n_bram_ports` 가 정한다. 포트를 늘리면 메모리 시간이
줄어 어느 지점에서 연산 아래로 내려가고, 그 순간부터 종단의 시간 이득이
사라진다. **이 실험은 그 교차점을 찾는다.**

★ 왜 별도 실험인가 ★
exp6(손익분기, 팀 1)은 `total_cycles` 즉 **연산축**만 본다. Fmax 저하를 곱해
"제어 논리를 붙일 값어치가 있는가" 를 묻는 실험이라 축이 다르다. 이쪽은
"메모리 절감이 시간으로 환산되는가" 를 묻는다. 둘은 보완 관계다.

★ 이 실험이 답하는 것 ★
  (1) 병목이 넘어가는 포트 수는 몇인가
  (2) 넘어간 뒤에도 남는 이득은 무엇인가 (읽기 = 동적 전력)
  (3) 문맥 길이가 교차점을 옮기는가
  (4) ★ Fmax 저하까지 넣으면 기준선(①)을 이기는가 — **메모리 포함 축에서**

exp6 은 total_cycles(연산축)로 손익분기를 본다. 비트평면 순차가 구조적으로 8배
불리한 축이라 ① 대비로는 절대 안 이긴다(0/81 조합). 그런데 ① 도 K 를 전부 읽어야
하므로 메모리를 넣고 같은 잣대로 재면 이야기가 달라진다. 이쪽이 그 축이다.

발견 경위 — `n_bram_ports` 를 2 -> 32 로 고쳐 "메모리가 연산의 11배" 인
불균형을 없앴는데, 그 결과 병목이 연산으로 넘어가 종단의 시간 이득이
1.00 배가 됐다. 고친 것은 맞지만 32 까지 갈 필요는 없었다.
-> ARCHITECTURE.md 4.1.1, current_state.md 8.1
"""

from __future__ import annotations

from dataclasses import replace

from src.decode_loop import run_decode
from utils.cost_model import ResourceReport, compare
from utils.io import save_records

from . import bram_from_config, build_workbench, load_config, spec_from_config

NAME = "exp7_memory_bottleneck"


def run(cfg=None, verbose: bool = True) -> list[dict]:
    cfg = cfg or load_config()
    sw = cfg.get("sweeps.exp7", {}) or {}
    seq_lens = sw.get("seq_len", [256, 512, 1024])
    ports = sw.get("n_bram_ports_sweep", [2, 4, 8, 16, 24, 32, 64])
    top_ks = sw.get("top_k", [16])
    margins = sw.get("margin", [0.0, 0.5])

    sched, bram = spec_from_config(cfg), bram_from_config(cfg)
    res = {d: ResourceReport.from_config(cfg, k) for d, k in
           (("baseline", "baseline"), ("seq", "seq_no_et"),
            ("exact", "exact_et"), ("approx", "approx_et"))}
    records: list[dict] = []

    for seq_len in seq_lens:
        wb = build_workbench(cfg, seq_len=int(seq_len))

        for n_ports in ports:
            b = replace(bram, n_ports=int(n_ports))

            # ② 종단 없는 비트평면 순차 — 종단의 순수 이득을 재려면 이게 기준이다.
            #    ① 은 비트평면조차 안 쓰므로 여기서는 축이 아니다 (exp6 이 본다).
            ref = run_decode(wb, design="seq", top_k=int(top_ks[0]), margin=0.0,
                             sched=sched, bram=b, keep_trace=False).summary
            # ① 기준선도 같은 포트로. 메모리 축에서는 ① 도 병목이라 함께 움직인다.
            base = run_decode(wb, design="baseline", top_k=int(top_ks[0]), margin=0.0,
                              sched=sched, bram=b, keep_trace=False).summary

            for top_k in top_ks:
                for margin in margins:
                    design = "exact" if margin == 0.0 else "approx"
                    s = run_decode(wb, design=design, top_k=int(top_k),
                                   margin=float(margin), sched=sched, bram=b,
                                   keep_trace=False).summary

                    # 병목은 max() 가 무엇을 집는지로 정해진다
                    bound = "memory" if s["total_memory_cycles"] > s["total_cycles"] else "compute"
                    # ★ Fmax 저하까지 반영한 실효 speedup — 메모리 포함 축에서
                    c1 = compare(res["baseline"], res[design],
                                 base["total_cycles_with_memory"],
                                 s["total_cycles_with_memory"])
                    c2 = compare(res["seq"], res[design],
                                 ref["total_cycles_with_memory"],
                                 s["total_cycles_with_memory"])
                    records.append({
                        "vs1_effective_speedup": c1["effective_speedup"],
                        "vs2_effective_speedup": c2["effective_speedup"],
                        "vs1_cycle_speedup": 1.0 / c1["cycle_ratio"] if c1["cycle_ratio"] else 0.0,
                        "baseline_total_cycles": base["total_cycles_with_memory"],
                        "seq_len": int(seq_len), "n_ports": int(n_ports),
                        "top_k": int(top_k), "margin": float(margin), "design": design,
                        "bottleneck": bound,
                        # 시간축 — 병목이 연산이면 1.0 근처로 붙는다
                        "time_ratio_vs_seq": s["total_cycles_with_memory"]
                                             / ref["total_cycles_with_memory"],
                        # 읽기축 — 병목과 무관하게 종단이 실제로 줄인 양
                        "read_ratio_vs_seq": (s["words_bram"] / ref["words_bram"])
                                             if ref["words_bram"] else 1.0,
                        "read_saving_bram": s["read_saving_bram"],
                        "compute_cycles": s["total_cycles"],
                        "memory_cycles": s["total_memory_cycles"],
                        "total_cycles": s["total_cycles_with_memory"],
                        "ref_total_cycles": ref["total_cycles_with_memory"],
                        "memory_bound_frac": s.get("memory_bound_frac", 0.0),
                        f"top{top_k}_retention": s.get(f"top{top_k}_retention", 1.0),
                    })

    save_records(records, NAME, cfg)
    if verbose:
        _report(records, cfg)
    return records


def crossover_ports(records: list[dict], seq_len: int, margin: float = 0.0) -> int | None:
    """병목이 memory -> compute 로 넘어가는 최소 포트 수.

    없으면 None (스윕 범위 안에서 안 넘어갔다는 뜻).
    """
    rows = sorted((r for r in records
                   if r["seq_len"] == seq_len and r["margin"] == margin),
                  key=lambda r: r["n_ports"])
    for r in rows:
        if r["bottleneck"] == "compute":
            return int(r["n_ports"])
    return None


def _report(records: list[dict], cfg) -> None:
    seq_lens = sorted({r["seq_len"] for r in records})
    margins = sorted({r["margin"] for r in records})

    for margin in margins:
        design = "exact" if margin == 0.0 else "approx"
        print(f"\n  [{design}, margin={margin}]  ② 대비 비율 — 1.0 미만이면 이득")
        print(f"    {'포트':>5s} |" + "".join(f"{'N=' + str(n):>21s}" for n in seq_lens))
        print(f"    {'':>5s} |" + "".join(f"{'시간':>8s}{'읽기':>7s}{'병목':>6s}" for _ in seq_lens))
        for n_ports in sorted({r["n_ports"] for r in records}):
            line = f"    {n_ports:>5d} |"
            for sl in seq_lens:
                m = [r for r in records
                     if r["n_ports"] == n_ports and r["seq_len"] == sl and r["margin"] == margin]
                if not m:
                    line += f"{'-':>21s}"
                    continue
                r = m[0]
                line += (f"{r['time_ratio_vs_seq']:>8.3f}{r['read_ratio_vs_seq']:>7.3f}"
                         f"{'메모리' if r['bottleneck'] == 'memory' else '연산':>6s}")
            print(line)

    print("\n  ★ Fmax 저하까지 반영한 실효 speedup — 메모리 포함 축, 채택 포트")
    adopted = max(r["n_ports"] for r in records)
    print(f"    (n_ports = {adopted})")
    print(f"    {'T':>6} {'설계':>8} {'margin':>7} {'vs① 사이클':>11} "
          f"{'vs① 실효':>10} {'vs② 실효':>10}  판정")
    for r in sorted((x for x in records if x["n_ports"] == adopted),
                    key=lambda x: (x["seq_len"], x["margin"])):
        v1 = r["vs1_effective_speedup"]
        print(f"    {r['seq_len']:>6} {r['design']:>8} {r['margin']:>7.2f} "
              f"{r['vs1_cycle_speedup']:>11.3f} {v1:>10.3f} "
              f"{r['vs2_effective_speedup']:>10.3f}  {'이긴다' if v1 > 1.0 else '진다'}")
    print("    ※ exp6 은 같은 값을 연산축(total_cycles)으로 낸다. 그쪽은 ① 대비로")
    print("      절대 안 이긴다 — 비트평면 순차가 그 축에서 구조적으로 8배 불리하다.")

    print("\n  ★ 교차점 (병목이 메모리 -> 연산 으로 넘어가는 최소 포트)")
    for sl in seq_lens:
        for margin in margins:
            c = crossover_ports(records, sl, margin)
            print(f"    N={sl:<5d} margin={margin:<4}  {c if c else '스윕 범위 내 없음'}")

    print("""
  읽는 법
    읽기 비율은 포트 수와 무관하다 — 종단 로직이 같으니 당연하다.
    시간 비율만 포트에 따라 움직인다. 병목이 '연산' 으로 바뀌는 순간
    시간 비율이 1.0 으로 붙고, 그때부터 메모리 절감은 시간이 아니라
    **동적 전력**으로만 남는다.

  설계 함의
    포트를 늘리면 절대 시간은 빨라지지만 종단의 값어치가 사라진다.
    둘 다 가질 수 없다. 어디에 설 것인지가 이 프로젝트의 결론이 된다.""")

    warns = cfg.provenance_warnings()
    if warns:
        print(f"\n  ⚠ config/hardware.yaml 에 추정치 {len(warns)}개 — Vivado 실측으로 교체 필요")


def main() -> int:
    from utils.io import enable_utf8_stdout

    enable_utf8_stdout()
    print(f"=== {NAME} ===")
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
