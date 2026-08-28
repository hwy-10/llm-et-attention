"""점수 손실 예산 (threshold score) — 무엇이 답을 얼마나 건드려도 되는가.

프로젝트의 원래 주장은 "조기 종단은 무손실"이었다. 실제로는 절감을 더 얻으려고
**의도적으로 손실을 감수하는 축**(``margin``)이 있으므로, "무손실"을 하나의
문장으로 두는 대신 **축마다 허용 손실을 명시**한다. 그것이 여기의 ``LossBudget`` 이다.

★ 축마다 예산이 다르다 ★
------------------------
::

    partials + bounds + ThetaPolicy   →  s_int, alive, term_plane   ← 정답
                    + margin           →  손실을 감수하는 유일한 축   예산 > 0
                    + decision_latency →  read_live                  예산 = 0
                    + schedule_policy  →  cycles, words_bram         예산 = 0

**스케줄 정책의 예산은 0 이다.** 정책은 회계만 바꾸므로 점수가 1 이라도 달라지면
그건 trade-off 가 아니라 버그다. 여기에 여유를 주면 그 버그를 숨기는 예산이 된다.
반대로 ``margin`` 은 여유를 주는 것이 설계 의도다.

지표 선택에 대하여
------------------
``spearman_rho`` 는 **쓰지 않는다.** 종단된 토큰은 ``-inf`` 로 마스킹되어 전부 동률이
되므로, 손실이 전혀 없는 exact 모드에서도 0.12 정도가 나온다. 손실 지표로 쓰면
멀쩡한 구현을 위반으로 잡는다 (:func:`test_spearman_is_not_a_loss_metric_here` 가
이 사실을 고정한다).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.accumulator import exact_int_scores, fold_and_quantize_query
from src.bounds import step_bounds
from src.designs import run_design
from src.masked_sum import partial_dots
from src.memory import BramSpec
from src.quantize import quantize_key, to_bitplanes
from src.schedule import POLICIES, ScheduleSpec
from utils.metrics import spearman_rho, top1_accuracy, topk_retention

TOP_K = 8
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)          # 고정 — 예산 수치는 이 집합에서 잰 것이다
SCHED = ScheduleSpec(lanes=32, batch_size=32, two_phase_split=3, compaction_cost_cycles=2)
BRAM = BramSpec(word_tokens=32, decision_latency_planes=1)
REF_POLICY = "none"                        # 정책 축의 기준점


# ---------------------------------------------------------------------------
# 예산 정의
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossBudget:
    """한 축에서 허용하는 손실의 상한."""

    axis: str
    topk_retention_min: float
    top1_acc_min: float
    max_abs_score_diff: int

    def check(self, m: dict) -> list[str]:
        """위반 항목을 문자열 목록으로 돌려준다 (빈 목록이면 예산 안)."""
        bad = []
        if m["topk_retention"] < self.topk_retention_min:
            bad.append(
                f"top{TOP_K} 보존율 {m['topk_retention']:.4f} < {self.topk_retention_min}"
            )
        if m["top1_acc"] < self.top1_acc_min:
            bad.append(f"top1 정확도 {m['top1_acc']:.4f} < {self.top1_acc_min}")
        if m["max_abs_score_diff"] > self.max_abs_score_diff:
            bad.append(
                f"최대 점수차 {m['max_abs_score_diff']} > {self.max_abs_score_diff}"
            )
        return bad


#: ★ 스케줄 정책의 예산은 0 이다 ★
#: 정책은 사이클과 읽기 회계만 바꾼다. 점수가 1이라도 달라지면 trade-off 가 아니라
#: 버그이므로 여유를 주지 않는다. 이 상수를 느슨하게 고치려는 사람은
#: test_policy_budget_must_stay_zero 를 먼저 지워야 한다 — 일부러 걸어 둔 문턱이다.
POLICY_BUDGET = LossBudget(
    axis="schedule policy",
    topk_retention_min=1.0,
    top1_acc_min=1.0,
    max_abs_score_diff=0,
)

#: margin 축 — **여기가 실제 trade-off 다.**
#: 아래 수치는 SEEDS 8개 평균 실측값에 약간의 여유를 둔 것이다 (합성 데이터 기준).
#: 실제 Llama 텐서 캡처가 들어오면 다시 잡아야 한다.
#:   margin  실측 top8 보존   예산
#:     0.00      1.0000       1.00
#:     0.02      0.9844       0.95
#:     0.05      0.9062       0.88
#:     0.10      0.9062       0.88
#:     0.15      0.8594       0.83
#:     0.25      0.7188       0.68
MARGIN_BUDGET = {
    0.00: 1.00,
    0.02: 0.95,
    0.05: 0.88,
    0.10: 0.88,
    0.15: 0.83,
    0.25: 0.68,
}


# ---------------------------------------------------------------------------
# 측정
# ---------------------------------------------------------------------------

def _case(seed: int, d: int = 64, T: int = 256):
    """한 디코드 스텝 + 그 스텝의 참값(전체 계산 점수)."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 1.0, size=(1, d))
    k = rng.normal(0, 1.0, size=(T, d))
    k[0] = 4.0 * (q[0] / np.linalg.norm(q[0])) * np.linalg.norm(k[0])
    key = quantize_key(k)
    fq = fold_and_quantize_query(q, key)
    p = partial_dots(fq.stored, to_bitplanes(key.stored, 8))[:, 0, :]
    return p, step_bounds(fq.stored[0]), exact_int_scores(fq.stored, key)[0]


def score_loss(ref_scores: np.ndarray, test_scores: np.ndarray,
               ref_raw: np.ndarray, test_raw: np.ndarray, k: int = TOP_K) -> dict:
    """두 점수 벡터 사이의 손실을 잰다.

    ``*_scores`` 는 마스킹된 실수 점수(순위용), ``*_raw`` 는 정수 점수(값 비교용)다.
    정수 쪽을 따로 보는 이유는, 순위가 같아도 값이 흔들리면 상류에서 무언가
    잘못된 것이기 때문이다.
    """
    ref2, test2 = np.asarray(ref_scores)[None, :], np.asarray(test_scores)[None, :]
    return {
        "topk_retention": topk_retention(ref2, test2, k),
        "top1_acc": top1_accuracy(ref2, test2),
        "max_abs_score_diff": int(np.abs(np.asarray(ref_raw, dtype=np.int64)
                                         - np.asarray(test_raw, dtype=np.int64)).max()),
    }


def _run(design, seed, policy, margin=0.0):
    p, b, truth = _case(seed)
    r = run_design(design, p, b, top_k=TOP_K, margin=margin, margin_mode="relative_gap",
                   schedule_policy=policy, sched=SCHED, bram=BRAM)
    return r, truth


# ---------------------------------------------------------------------------
# 정책 축 — 예산 0
# ---------------------------------------------------------------------------

def test_policy_budget_must_stay_zero():
    """★ 정책 축의 예산이 0 인지 자체를 고정한다.

    나중에 누군가 "조금은 봐 주자"며 이 상수를 올리면 여기서 막힌다.
    정책은 회계만 바꾸므로 손실이 생길 **이유가 없다** — 생겼다면 배선이 잘못된 것이지
    trade-off 가 아니다. 예산을 올리기 전에 왜 손실이 생기는지부터 밝혀야 한다.
    """
    assert POLICY_BUDGET.topk_retention_min == 1.0
    assert POLICY_BUDGET.top1_acc_min == 1.0
    assert POLICY_BUDGET.max_abs_score_diff == 0


@pytest.mark.parametrize("design,margin", [("exact", 0.0), ("approx", 0.15), ("seq", 0.0)])
@pytest.mark.parametrize("seed", SEEDS)
def test_policy_stays_within_budget(design, margin, seed):
    """정책을 바꿔도 점수가 예산(=0) 안에 있어야 한다.

    비교 기준은 **참값이 아니라 기준 정책** ``none`` 이다. 참값과 비교하면
    margin 이 만든 손실까지 섞여 들어와 정책 축을 따로 볼 수 없다.
    """
    ref, _ = _run(design, seed, REF_POLICY, margin)
    for policy in POLICIES:
        if policy == REF_POLICY:
            continue
        got, _ = _run(design, seed, policy, margin)
        m = score_loss(ref.scores, got.scores, ref.scores_raw, got.scores_raw)
        bad = POLICY_BUDGET.check(m)
        assert not bad, f"{design}/{policy}/seed{seed}: " + "; ".join(bad)


@pytest.mark.parametrize("design,margin", [("exact", 0.0), ("approx", 0.15)])
def test_policy_preserves_alive_and_read_live(design, margin):
    """점수뿐 아니라 **생존·종단평면·read_live 까지** 정책에 불변이어야 한다.

    기존 test_policy_does_not_change_scores 는 scores_raw 만 본다. 손실 허용
    스케줄러가 들어오면 점수는 그대로인데 alive/read_live 만 바뀔 수 있어
    그 테스트는 놓친다. 여기서 막는다.
    """
    for seed in SEEDS[:4]:
        ref, _ = _run(design, seed, REF_POLICY, margin)
        for policy in POLICIES:
            got, _ = _run(design, seed, policy, margin)
            np.testing.assert_array_equal(ref.alive, got.alive)
            np.testing.assert_array_equal(ref.term_plane, got.term_plane)
            np.testing.assert_array_equal(
                ref.step_result.read_live, got.step_result.read_live
            )


def test_budget_detects_a_lossy_scheduler(monkeypatch):
    """★ 예산 검사가 실제로 물리는지 — 위반을 한 번도 못 본 문턱은 믿을 수 없다.

    ``run_design`` 은 ``apply_schedule(sr, ...)`` 를 부른 **뒤에** ``sr.s_int`` 를 읽는다.
    따라서 스케줄러가 ``sr`` 을 제자리에서 건드리면 그대로 점수에 샌다 — 압축 구현이
    ``read_live`` 를 in-place 로 재배열하다 실수하면 나올 수 있는 실제 실패 모드다.
    그런 스케줄러를 일부러 주입하고, 위 예산 검사가 잡는지 확인한다.
    """
    import src.designs as designs

    real = designs.apply_schedule

    def lossy(sr, spec, bram, policy="compaction"):
        if policy == "compaction":          # 한 정책만 오염시킨다
            sr.s_int = sr.s_int.copy()
            sr.s_int[0] += 1                # 점수 1 만큼만 흔든다
            sr.alive = sr.alive.copy()
            sr.alive[-1] = not sr.alive[-1]
        return real(sr, spec, bram, policy)

    monkeypatch.setattr(designs, "apply_schedule", lossy)

    ref, _ = _run("exact", 0, REF_POLICY)
    got, _ = _run("exact", 0, "compaction")
    m = score_loss(ref.scores, got.scores, ref.scores_raw, got.scores_raw)

    assert POLICY_BUDGET.check(m), "the budget check failed to catch a tampered scheduler"
    assert m["max_abs_score_diff"] == 1
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(ref.alive, got.alive)


# ---------------------------------------------------------------------------
# margin 축 — 예산 > 0. 여기가 실제 trade-off 다
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("margin", sorted(MARGIN_BUDGET))
def test_margin_loss_within_budget(margin):
    """margin 을 키우면 손실이 늘지만 정해 둔 예산 안에 있어야 한다.

    이쪽 비교 대상은 **참값**이다 — 얼마나 틀렸는지를 보는 것이 목적이기 때문이다.
    """
    rets = []
    for seed in SEEDS:
        r, truth = _run("approx", seed, "compaction", margin)
        rets.append(topk_retention(truth[None, :], r.scores[None, :], TOP_K))
    mean = float(np.mean(rets))
    assert mean >= MARGIN_BUDGET[margin], (
        f"margin={margin}: top{TOP_K} retention {mean:.4f} < budget {MARGIN_BUDGET[margin]}"
    )


def test_margin_zero_is_lossless():
    """★ 경계 — margin 0 이면 손실이 정확히 0 이어야 한다.

    이게 깨지면 "조기 종단은 무손실"이라는 주장 자체가 무너진다.
    예산 표의 맨 윗줄이자 프로젝트의 마지노선이다.
    """
    for seed in SEEDS:
        r, truth = _run("exact", seed, "compaction", 0.0)
        assert topk_retention(truth[None, :], r.scores[None, :], TOP_K) == 1.0
        assert top1_accuracy(truth[None, :], r.scores[None, :]) == 1.0


def test_margin_loss_is_monotone_in_margin():
    """margin 을 키우면 보존율이 (약하게) 단조 감소해야 한다.

    손잡이가 뒤집혀 있으면 예산표 전체가 의미를 잃는다.
    """
    prev = 1.0 + 1e-9
    for margin in sorted(MARGIN_BUDGET):
        rets = [
            topk_retention(t[None, :], r.scores[None, :], TOP_K)
            for r, t in (_run("approx", s, "compaction", margin) for s in SEEDS)
        ]
        cur = float(np.mean(rets))
        assert cur <= prev + 1e-9, f"retention went up at margin={margin}"
        prev = cur


def test_margin_buys_something():
    """손실을 감수하면 **실제로 읽기가 줄어야** 한다.

    줄지 않는데 손실만 나면 그 margin 은 순손해다. 예산을 정당화하는 반대편이다.
    """
    words = {}
    for margin in (0.0, 0.25):
        w = [_run("approx", s, "compaction", margin)[0].reads.words_bram for s in SEEDS]
        words[margin] = float(np.mean(w))
    assert words[0.25] < words[0.0], f"reads did not drop as margin grew: {words}"


# ---------------------------------------------------------------------------
# 지표 선택 근거를 고정한다
# ---------------------------------------------------------------------------

def test_spearman_is_not_a_loss_metric_here():
    """★ spearman 을 예산에 넣으면 안 되는 이유를 고정한다.

    종단된 토큰은 -inf 로 마스킹되어 전부 동률이 된다. 그래서 **손실이 전혀 없는**
    exact 모드에서도 순위상관이 1.0 근처가 아니라 0.2 아래로 나온다.
    이걸 모르고 예산에 넣으면 멀쩡한 구현을 위반으로 잡는다.
    """
    r, truth = _run("exact", 0, "compaction", 0.0)

    assert topk_retention(truth[None, :], r.scores[None, :], TOP_K) == 1.0   # 무손실인데
    rho = spearman_rho(truth[None, :], r.scores[None, :])
    assert rho < 0.5, f"premise changed (rho={rho:.4f}); re-check the budget metric"
    assert np.isneginf(r.scores[~r.alive]).all()      # 그 원인: 종단 토큰 마스킹
