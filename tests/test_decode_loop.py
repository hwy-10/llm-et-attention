"""디코드 루프 검증.

고정 스냅샷이 아니라 T 가 자라는 루프를 도는 것이 이 프로젝트의 전제다.
"""

import numpy as np

from src.config import load_config
from src.dataset import synthetic_qk
from src.decode_loop import DecodeWorkbench, run_decode
from src.memory import BramSpec
from src.schedule import ScheduleSpec

SCHED = ScheduleSpec(lanes=32, compaction_cost_cycles=2)
BRAM = BramSpec(word_tokens=32, decision_latency_planes=1)


def _wb(seq_len=192, warmup=16, seed=0):
    snap = synthetic_qk(seq_len=seq_len, head_dim=64, seed=seed)
    return DecodeWorkbench.build(snap, warmup=warmup, seq_len=seq_len)


def test_context_grows_across_steps():
    wb = _wb()
    assert wb.step_tokens[0] == 16
    assert wb.step_tokens[-1] == 191
    assert np.all(np.diff(wb.step_tokens) > 0), "문맥이 스텝마다 자라야 한다"


def test_baseline_and_seq_are_lossless():
    wb = _wb()
    for design in ("baseline", "seq"):
        r = run_decode(wb, design=design, top_k=8, sched=SCHED, bram=BRAM, keep_trace=False)
        assert r.summary["top8_retention"] == 1.0
        assert r.summary["softmax_kl"] == 0.0
        assert r.summary["read_saving_ideal"] == 0.0


def test_exact_mode_preserves_topk_across_whole_loop():
    """★ 루프 전체에서 정확 모드가 무손실이어야 한다 ★"""
    for seed in (0, 1):
        wb = _wb(seed=seed)
        for top_k in (4, 8, 16):
            r = run_decode(wb, design="exact", top_k=top_k, sched=SCHED, bram=BRAM,
                           eval_top_k=(top_k,), keep_trace=False)
            assert r.summary[f"top{top_k}_retention"] == 1.0, (
                f"seed={seed} k={top_k}: 정확 모드가 top-k 를 잃었다"
            )


def test_exact_mode_no_excess_loss_vs_topk_oracle():
    """정확 모드는 참 top-k 보다 많은 토큰을 남기므로 오라클보다 정확해야 한다."""
    wb = _wb()
    r = run_decode(wb, design="exact", top_k=8, sched=SCHED, bram=BRAM, keep_trace=False)
    assert r.summary["kl_excess_from_termination"] <= 1e-9


def test_termination_actually_happens():
    """종단이 전혀 안 일어나면 프로젝트 전제가 무너진다 (가이드 8.3 1단계)."""
    wb = _wb()
    r = run_decode(wb, design="exact", top_k=8, sched=SCHED, bram=BRAM, keep_trace=False)
    assert r.summary["mean_term_plane"] < 8.0
    assert r.summary["read_saving_ideal"] > 0.0


def test_margin_increases_savings():
    wb = _wb()
    prev = -1.0
    for margin in (0.0, 0.1, 0.3, 0.6):
        r = run_decode(wb, design="approx", top_k=8, margin=margin,
                       sched=SCHED, bram=BRAM, keep_trace=False)
        cur = r.summary["read_saving_ideal"]
        assert cur >= prev - 1e-9, f"margin={margin} 에서 절감이 줄었다"
        prev = cur


def test_sound_theta_policies_are_lossless():
    """★ θ 가 이번 스텝의 하한(또는 참값)에서 나오면 무손실이어야 한다 ★"""
    wb = _wb()
    for pol in ("every_plane", "once_at_m", "oracle_fixed"):
        r = run_decode(wb, design="exact", top_k=8, theta_policy=pol, once_at_m=2,
                       sched=SCHED, bram=BRAM, keep_trace=False)
        assert r.summary["top8_retention"] == 1.0, f"{pol} 에서 top-k 손실 발생"


def test_prev_step_policy_is_not_sound():
    """★ 설계 발견: prev_step 은 margin=0 인데도 손실이 난다 ★

    θ 를 직전 디코드 스텝에서 가져오므로 이번 스텝의 하한과 무관하다.
    직전 θ 가 이번 스텝의 k번째 점수보다 크면 참 top-k 가 잘린다.
    부분 정렬 회로를 없애는 대신 무손실 보장을 포기하는 교환이다.
    """
    wb = _wb()
    r = run_decode(wb, design="exact", top_k=8, theta_policy="prev_step",
                   sched=SCHED, bram=BRAM, keep_trace=False)
    ret = r.summary["top8_retention"]
    assert ret < 1.0, "prev_step 이 무손실로 나왔다면 테스트 조건을 다시 볼 것"
    # 다만 손실이 파괴적이어서는 안 된다 (실용성이 남아 있는지 확인)
    assert ret > 0.5, f"prev_step 손실이 너무 크다: top8_retention={ret:.3f}"
    # 대신 종단은 더 일찍 일어나야 한다 (평면 0부터 판정 가능)
    ref = run_decode(wb, design="exact", top_k=8, theta_policy="every_plane",
                     sched=SCHED, bram=BRAM, keep_trace=False)
    assert r.summary["mean_term_plane"] <= ref.summary["mean_term_plane"] + 1e-9


def test_every_plane_terminates_at_least_as_much_as_once_at_m():
    wb = _wb()
    a = run_decode(wb, design="exact", top_k=8, theta_policy="every_plane",
                   sched=SCHED, bram=BRAM, keep_trace=False)
    b = run_decode(wb, design="exact", top_k=8, theta_policy="once_at_m", once_at_m=4,
                   sched=SCHED, bram=BRAM, keep_trace=False)
    assert a.summary["read_saving_ideal"] >= b.summary["read_saving_ideal"] - 1e-9


def test_trace_lengths_match():
    wb = _wb()
    r = run_decode(wb, design="exact", top_k=8, sched=SCHED, bram=BRAM, keep_trace=True)
    for key, arr in r.per_step.items():
        assert len(arr) == wb.n_steps, key


def test_config_driven_path_runs():
    """config/*.yaml 만으로 전체 경로가 도는지 (미니 YAML 파서 포함)."""
    cfg = load_config()
    assert cfg.head_dim == 64
    assert cfg.n_planes == 8
    from src.dataset import snapshot_from_config
    from src.decode_loop import workbench_from_config

    snap = snapshot_from_config(cfg, seq_len=96)
    wb = workbench_from_config(cfg, snap, seq_len=96, warmup=16)
    r = run_decode(wb, design="exact", top_k=4, keep_trace=False)
    assert r.summary["n_steps"] == 80
