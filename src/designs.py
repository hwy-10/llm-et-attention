"""★ 네 비교 설계를 하나의 인터페이스로 ★ — 배경지식 가이드 8.1절

    ① baseline : 일반적인 병렬 INT8 곱셈 누산 구조        (기준선, DSP 사용)
    ② seq      : 비트평면 순차 처리, 조기 종단 없음        (순차 전환 비용 분리)
    ③ exact    : 비트평면 순차 + 정확 모드 종단            (무손실 순수 이득)
    ④ approx   : 비트평면 순차 + 근사 모드 종단            (절감–정확도 곡선)

★ ②와 ③의 비교가 승부처다 ★
②에서 순차 처리로 전환할 때 치르는 비용을 재고, ③의 종단 이득이 그 비용을
넘는지 확인하는 구조다.

불변식 (tests/test_designs.py 가 검증):
    ①, ②, ③ 의 점수는 **비트 단위로 완전히 동일**해야 한다.
    ③ 은 무손실 판정이므로 종단을 해도 top-k 집합이 바뀌지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .accumulator import accumulate
from .bounds import StepBounds
from .memory import BramSpec, ReadAccount
from .schedule import ScheduleResult, ScheduleSpec, apply as apply_schedule, baseline_cycles
from .terminator import StepResult, masked_scores, run_step
from .threshold import ThetaPolicy

DESIGNS = ("baseline", "seq", "exact", "approx")

# 설계별 기본 성격
_TERMINATION = {"baseline": False, "seq": False, "exact": True, "approx": True}
_DEFAULT_SCHEDULE = {"baseline": "none", "seq": "none", "exact": "compaction", "approx": "compaction"}


@dataclass
class DesignResult:
    """설계 하나가 한 디코드 스텝에서 낸 결과."""

    design: str
    scores: np.ndarray            # (n_active,) 정수 점수 (종단 토큰은 -inf 마스킹)
    scores_raw: np.ndarray        # (n_active,) 마스킹 전 값
    alive: np.ndarray
    term_plane: np.ndarray
    cycles: int
    reads: ReadAccount
    schedule: ScheduleResult | None = None
    step_result: StepResult | None = None
    extra: dict = field(default_factory=dict)

    @property
    def mean_term_plane(self) -> float:
        return float(self.term_plane.mean()) if self.term_plane.size else 0.0

    @property
    def survivor_frac(self) -> float:
        return float(self.alive.mean()) if self.alive.size else 1.0


def run_design(
    design: str,
    partials: np.ndarray,
    bounds: StepBounds,
    *,
    top_k: int = 8,
    margin: float = 0.0,
    margin_mode: str = "relative_width",
    theta_policy: str = "every_plane",
    once_at_m: int = 3,
    schedule_policy: str | None = None,
    sched: ScheduleSpec | None = None,
    bram: BramSpec | None = None,
    prev_theta_int: float | None = None,
    oracle_theta_int: float | None = None,
) -> DesignResult:
    """설계 하나를 한 스텝에 대해 실행한다.

    Parameters
    ----------
    partials : (n_planes, n_active) — 이 스텝의 P_b (MSB 우선)
    bounds   : 이 스텝의 Q+ / Q−
    """
    if design not in DESIGNS:
        raise KeyError(f"unknown design {design!r}; choose from {DESIGNS}")
    sched = sched or ScheduleSpec()
    bram = bram or BramSpec()
    p = np.asarray(partials, dtype=np.int64)
    n_planes, n_active = p.shape

    eff_margin = margin if design == "approx" else 0.0
    policy = ThetaPolicy(
        name=theta_policy,
        top_k=top_k,
        once_at_m=once_at_m,
        margin=eff_margin,
        margin_mode=margin_mode,
    )

    sr = run_step(
        p,
        bounds,
        policy,
        decision_latency=bram.decision_latency_planes,
        prev_theta=prev_theta_int,
        oracle_theta=oracle_theta_int,
        enable_termination=_TERMINATION[design],
    )

    pol = schedule_policy or _DEFAULT_SCHEDULE[design]
    schedule = apply_schedule(sr, sched, bram, policy=pol)

    if design == "baseline":
        # 병렬 INT8 MAC — 비트평면 순차가 아니므로 사이클 모델이 다르다.
        # 읽는 비트 총량은 동일(K 전체)하므로 읽기 회계는 dense 를 그대로 쓴다.
        cycles = baseline_cycles(n_active, sched)
        scores_raw = accumulate(p)
        return DesignResult(
            design=design,
            scores=scores_raw.astype(np.float64),
            scores_raw=scores_raw,
            alive=np.ones(n_active, dtype=bool),
            term_plane=np.full(n_active, n_planes, dtype=np.int32),
            cycles=cycles,
            reads=schedule.reads,
            schedule=schedule,
            step_result=sr,
            extra={"uses_dsp": True},
        )

    return DesignResult(
        design=design,
        scores=masked_scores(sr),
        scores_raw=sr.s_int,
        alive=sr.alive,
        term_plane=sr.term_plane,
        cycles=schedule.cycles,
        reads=schedule.reads,
        schedule=schedule,
        step_result=sr,
        extra={"uses_dsp": False, "theta_final": sr.extra.get("theta_final")},
    )


def design_label(design: str) -> str:
    return {
        "baseline": "① 병렬 INT8 MAC (기준선)",
        "seq": "② 비트평면 순차 (종단 없음)",
        "exact": "③ + 정확 모드 종단",
        "approx": "④ + 근사 모드 종단",
    }[design]
