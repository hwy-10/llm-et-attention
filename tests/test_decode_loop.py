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
    assert np.all(np.diff(wb.step_tokens) > 0), "context must grow at every step"


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
                f"seed={seed} k={top_k}: exact mode lost top-k"
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
        assert cur >= prev - 1e-9, f"savings dropped at margin={margin}"
        prev = cur


def test_sound_theta_policies_are_lossless():
    """★ θ 가 이번 스텝의 하한(또는 참값)에서 나오면 무손실이어야 한다 ★"""
    wb = _wb()
    for pol in ("every_plane", "once_at_m", "oracle_fixed"):
        r = run_decode(wb, design="exact", top_k=8, theta_policy=pol, once_at_m=2,
                       sched=SCHED, bram=BRAM, keep_trace=False)
        assert r.summary["top8_retention"] == 1.0, f"top-k loss under policy {pol}"


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
    assert ret < 1.0, "prev_step came out lossless; re-check the test setup"
    # 다만 손실이 파괴적이어서는 안 된다 (실용성이 남아 있는지 확인)
    assert ret > 0.5, f"prev_step loss too large: top8_retention={ret:.3f}"
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


# ---------------------------------------------------------------------------
# ★ 2026-08-28 돌연변이 시험에서 드러난 구멍 (팀 2 신규 배정)
# ---------------------------------------------------------------------------
def test_oracle_theta_is_exactly_the_kth_true_score():
    """★ oracle_fixed 의 θ 는 '이번 스텝 참 점수의 정확히 k번째'여야 한다.

    `np.partition(ex, -kk)[-kk]` 를 `[-kk-1]` 로 바꿔도 기존 테스트가 안 잡았다.
    θ 가 낮아지면 종단이 줄 뿐 무손실은 유지되므로 **보존율로는 안 보인다.**
    이 참조선이 틀리면 '오라클 대비 몇 %' 라는 비교가 통째로 어긋난다.

    관측 가능한 불변식으로 잡는다.

        θ_oracle      = 참 점수 k번째
        θ_every_plane = 하한 k번째  <=  참 점수 k번째

    따라서 **oracle 이 건전한 정책 중 종단 상한선**이어야 한다. θ 를 k+1 번째로
    낮추면 종단이 줄어 생존이 늘고 절감이 떨어진다.
    """
    wb = _wb(seq_len=192, warmup=16)
    k = 8
    o = run_decode(wb, design="exact", top_k=k, theta_policy="oracle_fixed",
                   sched=SCHED, bram=BRAM, keep_trace=False).summary
    e = run_decode(wb, design="exact", top_k=k, theta_policy="every_plane",
                   sched=SCHED, bram=BRAM, keep_trace=False).summary

    assert o["read_saving_ideal"] > e["read_saving_ideal"], (
        f"oracle {o['read_saving_ideal']:.4f} 이 every_plane "
        f"{e['read_saving_ideal']:.4f} 보다 커야 한다 — θ 가 k번째가 아니다"
    )
    assert o["mean_survivor_frac"] < e["mean_survivor_frac"], "oracle 이 더 많이 종단해야 한다"
    assert o[f"top{k}_retention"] == 1.0, "oracle 은 건전한 θ 라 무손실이어야 한다"

    # θ 자체도 직접 확인 — 참 점수 중 k개 이상이 θ 이상, k개 미만이 θ 초과
    s = 30
    n = int(wb.step_tokens[s])
    ex = wb.exact_int[s, :n]
    kk = min(k, n)
    want = float(np.partition(ex, -kk)[-kk])
    assert int(np.sum(ex >= want)) >= kk
    assert int(np.sum(ex > want)) < kk


def test_retention_denominator_uses_min_k_and_active():
    """★ 보존율 분모는 `min(k, n_active)` 여야 한다.

    `/ kk` 를 `/ k` 로 바꿔도 기존 테스트가 안 잡았다. 활성 토큰이 k 보다
    많은 경우만 돌렸기 때문이다. 활성 토큰이 k 보다 적으면 분모가 부풀어
    **보존율이 실제보다 낮게** 나온다.

    warmup=4 로 두면 첫 스텝의 활성 토큰이 4개뿐이라 k=16 과 어긋난다.
    """
    wb = _wb(seq_len=64, warmup=4)
    assert int(wb.step_tokens[0]) < 16, "이 테스트의 전제 — 첫 스텝 활성 토큰 < k"

    # ★ 평가 k 와 설계 top_k 를 같게 둔다.
    #   설계는 top_k 개 보존만 보장한다. top_k=8 인 실행에서 top-16 을 평가하면
    #   9~16위가 종단되는 것이 **정상**이라 분모 문제와 구분이 안 된다.
    r = run_decode(wb, design="exact", top_k=16, sched=SCHED, bram=BRAM,
                   eval_top_k=(16,), keep_trace=False)
    # 정확 모드는 무손실이므로, 분모가 min(k, n_active) 이면 1.0 이 나온다.
    # `/ k` 로 바꾸면 첫 스텝이 4/16 = 0.25 가 되어 평균이 내려간다.
    assert r.summary["top16_retention"] == 1.0, (
        f"보존율 {r.summary['top16_retention']:.4f} — 분모가 min(k, n_active) 가 아니다"
    )


# ---------------------------------------------------------------------------
# §6 통합부 정밀 검증 (팀 2 신규 배정)
# ---------------------------------------------------------------------------
def test_prev_theta_roundtrip_matches_to_real_scores():
    """★ prev_step 의 정확성이 이 변환 하나에 걸려 있다.

        to_real_scores :  s_real = (s_int − corr) · scale / sqrt(d)
        decode_loop    :  s_int  =  s_real · sqrt(d) / scale + corr

    두 식이 서로의 역이어야 한다. 어긋나면 직전 스텝 θ 가 엉뚱한 값으로
    이번 스텝에 들어가고, prev_step 의 손실률이 통째로 틀린다.
    """
    wb = _wb(seq_len=128, warmup=16)
    for s in range(0, wb.n_steps, 7):
        n = int(wb.step_tokens[s])
        scale = float(np.asarray(wb.fq.scale).reshape(-1)[s])
        zp = float(wb.fq.zp_correction[s])
        s_int = wb.exact_int[s, :n].astype(np.float64)
        s_real = wb.real_scores(s, s_int)
        back = s_real * np.sqrt(wb.head_dim) / scale + zp      # decode_loop:190 의 식
        assert np.abs(back - s_int).max() < 1e-6, f"step {s}"


def test_real_scores_agrees_with_accumulator():
    """`DecodeWorkbench.real_scores` 가 `accumulator.to_real_scores` 를 다시 구현한다.

    두 벌이 갈라지면 요약 CSV 와 정확도 지표가 다른 말을 하게 된다.
    """
    from src.accumulator import FoldedQuery, to_real_scores

    wb = _wb(seq_len=128, warmup=16)
    for s in range(0, wb.n_steps, 7):
        n = int(wb.step_tokens[s])
        si = wb.exact_int[s, :n].astype(np.float64)
        fq1 = FoldedQuery(stored=wb.fq.stored[s:s + 1],
                          scale=np.asarray(wb.fq.scale).reshape(-1)[s:s + 1],
                          zp_correction=wb.fq.zp_correction[s:s + 1])
        np.testing.assert_allclose(wb.real_scores(s, si), to_real_scores(si, fq1, wb.head_dim))
    # 종단 토큰(-inf)도 양쪽이 같게 다뤄야 한다
    si = np.array([100.0, -np.inf, 300.0])
    assert np.isneginf(wb.real_scores(0, si)[1])


def test_recorded_seq_len_is_one_less_than_requested():
    """★ 알려진 불일치 — 기록되는 `seq_len` 이 요청값보다 1 작다.

    `step_tokens` 는 토큰 인덱스라 마지막이 `seq_len − 1` 이다. causal 이라
    인덱스 == 활성 토큰 수이므로 계산은 맞지만, **CSV 에 실리는 이름이 오해를 부른다.**

    `crosscheck` 는 `(design, seq_len, top_k, margin)` 으로 대조하는데,
    `predict_for()` 가 실측 키의 seq_len 을 다시 붙여 주는 덕분에 지금은 통과한다.
    RTL 팀이 `outputs/raw/*.csv` 를 보고 시뮬레이션 조건을 정하면 어긋난다.

    이 테스트는 **현재 동작을 고정**한다. 이름을 고치기로 하면 뒤집을 것.
    """
    wb = _wb(seq_len=128, warmup=16)
    r = run_decode(wb, design="exact", top_k=8, sched=SCHED, bram=BRAM, keep_trace=False)
    assert r.config["seq_len"] == 127, "요청 128 -> 기록 127"
    assert int(wb.step_tokens[-1]) == 127


def test_kl_excess_is_negative_in_exact_mode():
    """★ 이름이 오해를 부른다 — `excess` 인데 0 이하다.

    정확 모드는 top-k 보다 **많은** 토큰을 남기므로 하드 top-k 오라클보다 정확하다.
    따라서 '종단이 추가로 유발한 손실'은 음수다. 0 이상이 나오면 종단이 오라클보다
    나쁘다는 뜻이라 설계 전제가 무너진 것이다.
    """
    wb = _wb(seq_len=128, warmup=16)
    m = run_decode(wb, design="exact", top_k=8, sched=SCHED, bram=BRAM, keep_trace=False).summary
    assert m["kl_excess_from_termination"] <= 1e-9
    assert m["softmax_kl"] < m["softmax_kl_topk_oracle"]


def test_baseline_and_seq_have_no_read_saving():
    """설계 ①·② 는 종단이 없으므로 읽기 절감이 정확히 0 이어야 한다."""
    from src.designs import run_design

    wb = _wb(seq_len=128, warmup=16)
    s = 5
    n = int(wb.step_tokens[s])
    p, b = wb.partials[:, s, :n], wb.bounds_at(s)
    # ★ WORD_TOKENS=1 (2026-08-28 확정값)을 쓴다.
    #   word_tokens=32 로 두면 이 스텝의 토큰 21개가 한 워드에 다 들어가
    #   종단이 일어나도 워드 절감이 0 이 된다 — BRAM 워드폭 함정 그 자체다.
    bram1 = BramSpec(word_tokens=1, word_bits=64, decision_latency_planes=1)
    for design in ("baseline", "seq"):
        r = run_design(design, p, b, top_k=8, sched=SCHED, bram=bram1)
        assert r.reads.words_bram == r.reads.words_dense, design
        assert r.reads.bram_saving == 0.0, design
    ex = run_design("exact", p, b, top_k=8, sched=SCHED, bram=bram1)
    assert ex.reads.bram_saving > 0.0, "정확 모드는 절감이 있어야 한다"

    # 같은 조건에서 word_tokens 를 키우면 절감이 사라진다 (함정의 실증)
    wide = run_design("exact", p, b, top_k=8, sched=SCHED,
                      bram=BramSpec(word_tokens=32, decision_latency_planes=1))
    assert wide.reads.bram_saving == 0.0, "토큰 21개가 32짜리 워드 하나에 들어간다"
