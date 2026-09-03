"""EXP7 검증 — 병목 위치가 종단의 가치를 정한다는 주장이 코드로 성립하는가.

이 실험의 결론은 세 가지 성질에 기대고 있다. 셋 다 여기서 고정한다.

  [1] 읽기 절감은 포트 수와 무관하다        (종단 로직이 같으므로)
  [2] 포트를 늘리면 병목이 메모리 -> 연산 으로 한 방향으로만 넘어간다
  [3] 넘어간 뒤에는 시간 비율이 1.0 위로 붙는다 (= 종단이 시간을 못 산다)
"""

from dataclasses import replace

import numpy as np

from experiments.exp7_memory_bottleneck import crossover_ports, run
from src.config import load_config
from src.dataset import snapshot_from_config
from src.decode_loop import run_decode, workbench_from_config
from src.schedule import bram_from_config, spec_from_config


def _wb(cfg, seq_len=256, seed=0):
    return workbench_from_config(cfg, snapshot_from_config(cfg, seed=seed, seq_len=seq_len))


def _sweep(seq_len=256, ports=(2, 8, 16, 24, 32, 64), top_k=16, margin=0.0):
    cfg = load_config()
    wb = _wb(cfg, seq_len)
    sched, bram = spec_from_config(cfg), bram_from_config(cfg)
    out = []
    for p in ports:
        b = replace(bram, n_ports=p)
        ref = run_decode(wb, design="seq", top_k=top_k, margin=0.0,
                         sched=sched, bram=b, keep_trace=False).summary
        s = run_decode(wb, design="exact" if margin == 0 else "approx",
                       top_k=top_k, margin=margin, sched=sched, bram=b,
                       keep_trace=False).summary
        out.append((p, s, ref))
    return out


# --- [1] 읽기 절감은 포트와 무관하다 ---------------------------------------

def test_read_saving_does_not_depend_on_ports():
    """포트는 '한 사이클에 몇 워드를 읽나' 일 뿐, '몇 워드를 읽나' 가 아니다.

    이게 깨지면 memory_cycles 와 words_bram 이 뒤섞인 것이다.
    """
    rows = _sweep()
    words = {s["words_bram"] for _p, s, _r in rows}
    assert len(words) == 1, f"포트를 바꿨는데 읽은 워드 수가 달라졌다: {words}"
    savings = {round(s["read_saving_bram"], 9) for _p, s, _r in rows}
    assert len(savings) == 1


def test_memory_cycles_scale_inversely_with_ports():
    """memory_cycles = ceil(words / n_ports) — 포트를 2배로 하면 절반이 된다."""
    rows = {p: s for p, s, _r in _sweep(ports=(2, 4, 8, 16))}
    for lo, hi in ((2, 4), (4, 8), (8, 16)):
        a, b = rows[lo]["total_memory_cycles"], rows[hi]["total_memory_cycles"]
        assert abs(a / b - 2.0) < 0.05, f"n_ports {lo}->{hi} 에서 {a}/{b} = {a / b:.3f}"


# --- [2] 병목은 한 방향으로만 넘어간다 --------------------------------------

def test_bottleneck_flips_once_and_only_one_way():
    rows = _sweep()
    bound = ["memory" if s["total_memory_cycles"] > s["total_cycles"] else "compute"
             for _p, s, _r in rows]
    # memory* compute*  — 한 번만 바뀌어야 한다
    flips = sum(1 for a, b in zip(bound, bound[1:]) if a != b)
    assert flips <= 1, f"병목이 여러 번 뒤집혔다: {bound}"
    if flips == 1:
        assert bound[0] == "memory" and bound[-1] == "compute", bound


def test_crossover_exists_and_is_reported():
    recs = [{"seq_len": 256, "margin": 0.0, "n_ports": p,
             "bottleneck": "memory" if p < 24 else "compute"}
            for p in (2, 8, 16, 24, 32)]
    assert crossover_ports(recs, 256, 0.0) == 24
    none = [dict(r, bottleneck="memory") for r in recs]
    assert crossover_ports(none, 256, 0.0) is None


def test_compute_cycles_do_not_depend_on_ports():
    """연산축은 포트와 무관해야 한다. 섞이면 교차점 자체가 무의미해진다."""
    rows = _sweep()
    assert len({s["total_cycles"] for _p, s, _r in rows}) == 1


# --- [3] 넘어간 뒤에는 시간 이득이 사라진다 ---------------------------------

def test_time_ratio_rises_toward_and_past_one_as_ports_grow():
    rows = _sweep()
    ratios = [(p, s["total_cycles_with_memory"] / r["total_cycles_with_memory"])
              for p, s, r in rows]
    vals = [v for _p, v in ratios]
    # 단조 증가 (포트가 늘수록 종단의 시간 이득이 줄어든다)
    assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:])), ratios
    # 포트가 적을 때는 이득이 있고, 많을 때는 없다
    assert vals[0] < 0.95, f"포트 2 에서 시간 이득이 없다: {vals[0]:.3f}"
    assert vals[-1] > 0.99, f"포트 64 에서도 시간 이득이 남는다: {vals[-1]:.3f}"


def test_saving_persists_on_the_read_axis_even_when_time_gain_is_gone():
    """★ 이 실험의 결론. 시간 이득이 사라져도 읽기 절감은 그대로 남는다.

    그래서 연산 병목 영역에서 종단의 값어치는 '시간' 이 아니라 '전력' 이다.
    """
    rows = _sweep(ports=(64,))
    _p, s, r = rows[0]
    assert s["total_cycles_with_memory"] / r["total_cycles_with_memory"] > 0.99
    assert s["words_bram"] / r["words_bram"] < 0.95


def test_longer_context_pushes_the_crossover_higher():
    """문맥이 길수록 메모리 병목 구간이 넓다 = 종단이 더 오래 값어치를 한다."""
    def ratio_at(seq_len, p=24):
        _p, s, r = _sweep(seq_len=seq_len, ports=(p,))[0]
        return s["total_cycles_with_memory"] / r["total_cycles_with_memory"]
    assert ratio_at(512) < ratio_at(256), "긴 문맥이 더 불리하게 나왔다"


# --- 실험 자체가 도는가 -----------------------------------------------------

def test_run_produces_records_with_required_fields():
    cfg = load_config()
    cfg.sweeps["exp7"] = {
        "n_bram_ports_sweep": [8, 32], "seq_len": [128], "top_k": [16], "margin": [0.0],
    }
    recs = run(cfg, verbose=False)
    assert len(recs) == 2
    need = {"seq_len", "n_ports", "bottleneck", "time_ratio_vs_seq",
            "read_ratio_vs_seq", "compute_cycles", "memory_cycles"}
    for r in recs:
        assert need <= set(r), sorted(need - set(r))
        assert r["bottleneck"] in ("memory", "compute")
        assert np.isfinite(r["time_ratio_vs_seq"])
