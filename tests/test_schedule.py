"""스케줄 정책 검증 — 배경지식 가이드 6.3-(2)

정책은 사이클과 읽기를 바꾸지만 **점수는 절대 바꾸지 않는다**.
"""

import numpy as np

from src.accumulator import exact_int_scores, fold_and_quantize_query
from src.bounds import step_bounds
from src.designs import run_design
from src.masked_sum import partial_dots
from src.memory import BramSpec
from src.quantize import quantize_key, to_bitplanes
from src.schedule import POLICIES, ScheduleSpec, baseline_cycles, dense_cycles


def _case(seed=0, d=64, T=256):
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 1.0, size=(1, d))
    k = rng.normal(0, 1.0, size=(T, d))
    k[0] = 4.0 * (q[0] / np.linalg.norm(q[0])) * np.linalg.norm(k[0])
    key = quantize_key(k)
    fq = fold_and_quantize_query(q, key)
    p = partial_dots(fq.stored, to_bitplanes(key.stored, 8))[:, 0, :]
    return p, step_bounds(fq.stored[0]), exact_int_scores(fq.stored, key)[0]


SCHED = ScheduleSpec(lanes=32, batch_size=32, two_phase_split=3, compaction_cost_cycles=2)
BRAM = BramSpec(word_tokens=32, decision_latency_planes=1)


def test_policy_does_not_change_scores():
    """★ 스케줄은 회계만 바꾼다. 점수는 불변이어야 한다 ★"""
    p, b, _ = _case(0)
    ref = None
    for pol in POLICIES:
        r = run_design("exact", p, b, top_k=8, schedule_policy=pol, sched=SCHED, bram=BRAM)
        if ref is None:
            ref = r.scores_raw
        else:
            np.testing.assert_array_equal(ref, r.scores_raw)


def test_dense_and_baseline_cycles():
    # 토큰 256, lanes 32 -> 평면당 8사이클, 8평면 -> 64
    assert dense_cycles(256, 8, SCHED) == 64
    # 기준 설계: PE 32개, 토큰당 1사이클 -> 8사이클
    assert baseline_cycles(256, SCHED) == 8


def test_bitserial_is_structurally_slower_than_baseline():
    """★ 정직하게 짚어야 할 사실 ★

    비트평면 순차 처리는 8사이클을 쓰므로 기준 설계보다 사이클이 많다.
    제안의 근거는 사이클이 아니라 DSP 미사용 + 메모리 읽기 감소다.
    """
    assert dense_cycles(256, 8, SCHED) > baseline_cycles(256, SCHED)


def test_compaction_beats_batch_on_reads():
    """압축이 흩어진 배치보다 워드 읽기가 적어야 한다."""
    p, b, _ = _case(1)
    rb = run_design("exact", p, b, top_k=8, schedule_policy="batch", sched=SCHED, bram=BRAM)
    rc = run_design("exact", p, b, top_k=8, schedule_policy="compaction", sched=SCHED, bram=BRAM)
    assert rc.reads.words_bram <= rb.reads.words_bram


def test_none_policy_has_no_savings():
    p, b, _ = _case(2)
    r = run_design("seq", p, b, schedule_policy="none", sched=SCHED, bram=BRAM)
    assert r.reads.ideal_saving == 0.0
    assert r.reads.bram_saving == 0.0


def test_two_phase_reads_all_in_phase1():
    """2단계 처리는 1단계에서 전체를 읽으므로 그만큼 절감이 없다."""
    p, b, _ = _case(3)
    sched = ScheduleSpec(lanes=32, two_phase_split=3, compaction_cost_cycles=2)
    r = run_design("exact", p, b, top_k=8, schedule_policy="two_phase", sched=sched, bram=BRAM)
    n_tokens = p.shape[1]
    words_full = BRAM.n_words(n_tokens)
    assert r.reads.words_bram >= 3 * words_full


def test_pipeline_efficiency_bounded():
    p, b, _ = _case(4)
    for pol in POLICIES:
        r = run_design("exact", p, b, top_k=8, schedule_policy=pol, sched=SCHED, bram=BRAM)
        assert 0.0 <= r.schedule.pipeline_efficiency <= 1.0 + 1e-9
