"""조기 종단 제어기 — 배경지식 가이드 5.4 / 6.2절

한 디코드 스텝(쿼리 q 하나 + 활성 토큰 T개)을 MSB 평면부터 순차 처리하며
종단 판정을 수행한다.

    평면 m 처리 후:   S_m,  L_m = S_m + (2^(8-m)-1)·Q−,  U_m = S_m + (2^(8-m)-1)·Q+
    θ = 활성 토큰 하한 중 k번째로 큰 값
    U_m < θ + margin  ->  종단

★ 판정 지연 (decision_latency_planes) ★
평면 m 의 판정은 평면 m 의 가산 트리가 끝나야 나온다. 그 시점에 평면 m+1 의
읽기 요청은 이미 나갔을 수 있다. 따라서 평면 t 를 읽을 때 쓸 수 있는 마스크는
live[t − latency] 이다. 이 값을 0으로 두면 읽기 절감이 과대평가된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bounds import StepBounds
from .quantize import N_PLANES, plane_weights
from .threshold import ThetaPolicy, ThetaTracker, topk_indices


@dataclass
class StepResult:
    """디코드 스텝 하나의 결과."""

    s_int: np.ndarray            # (n_tokens,) 최종 정수 점수 (종단 토큰은 확정 시점 값)
    alive: np.ndarray            # (n_tokens,) bool — 끝까지 살아남은 토큰
    term_plane: np.ndarray       # (n_tokens,) int — 종단된 평면 수 (미종단이면 n_planes)
    read_live: np.ndarray        # (n_planes, n_tokens) bool — 평면별 실제 읽는 토큰
    theta_trace: np.ndarray      # (n_planes,) float
    live_count: np.ndarray       # (n_planes,) int — 평면별 논리적 활성 수
    n_planes: int = N_PLANES
    n_active: int = 0            # 이 스텝의 활성 토큰 수 (causal)
    extra: dict = field(default_factory=dict)

    @property
    def mean_term_plane(self) -> float:
        return float(self.term_plane.mean()) if self.term_plane.size else float(self.n_planes)

    @property
    def survivor_frac(self) -> float:
        return float(self.alive.mean()) if self.alive.size else 1.0


def run_step(
    partials: np.ndarray,
    bounds: StepBounds,
    policy: ThetaPolicy,
    *,
    decision_latency: int = 0,
    prev_theta: float | None = None,
    oracle_theta: float | None = None,
    enable_termination: bool = True,
) -> StepResult:
    """한 디코드 스텝을 MSB 평면부터 처리한다.

    Parameters
    ----------
    partials : (n_planes, n_tokens) int — 이 스텝의 P_b. index 0 = MSB.
    bounds   : 이 스텝의 Q+ / Q− (q 에만 의존)
    policy   : θ 정책 + 근사 마진
    decision_latency : 판정이 읽기에 반영되기까지의 평면 수
    enable_termination : False 면 설계 ② (순차 처리, 종단 없음)
    """
    p = np.asarray(partials, dtype=np.int64)
    if p.ndim != 2:
        raise ValueError(f"partials must be (n_planes, n_tokens), got {p.shape}")
    n_planes, n_tokens = p.shape
    weights = plane_weights(n_planes)

    s_m = np.zeros(n_tokens, dtype=np.int64)
    frozen = np.zeros(n_tokens, dtype=np.int64)
    live = np.ones(n_tokens, dtype=bool)
    term_plane = np.full(n_tokens, n_planes, dtype=np.int32)

    live_history: list[np.ndarray] = [live.copy()]   # live_history[m] = m개 처리 후
    theta_trace = np.full(n_planes, -np.inf, dtype=np.float64)
    live_count = np.zeros(n_planes, dtype=np.int64)
    read_live = np.zeros((n_planes, n_tokens), dtype=bool)

    tracker = ThetaTracker(policy, prev_theta=prev_theta, oracle_theta=oracle_theta)

    for t in range(n_planes):
        # 평면 t 를 읽을 때 사용할 수 있는 마스크 (판정 지연 반영)
        mask_idx = max(0, t - decision_latency)
        read_live[t] = live_history[mask_idx]
        live_count[t] = int(live.sum())

        # 부분 내적 누산 (죽은 토큰은 동결)
        s_m = s_m + weights[t] * p[t] * live

        m = t + 1
        if not enable_termination or m >= n_planes:
            live_history.append(live.copy())
            theta_trace[t] = tracker.theta
            continue

        lower = s_m + bounds.l_offset(m)
        upper = s_m + bounds.r(m)

        theta = tracker.update(lower, live, m)
        theta_trace[t] = theta

        if np.isfinite(theta):
            # gap = 1등 하한 − θ. 근사 마진의 기본 정규화 단위 (threshold.py 참조)
            gap = float(np.max(np.where(live, lower, -np.inf)) - theta) if np.any(live) else 0.0
            margin_abs = policy.margin_abs(bounds.width(m), bounds.q_pos, gap)
            kill = live & (upper < theta + margin_abs)
            survivors = live & ~kill

            # 최소 top_k 개는 남긴다.
            # ★ 전부-아니면-전무로 건너뛰면 안 된다 ★ 큰 margin 에서 가지치기가
            #   통째로 무효화되어 근사 모드가 공격적으로 동작하지 않는다.
            #   상한이 큰 순으로 top_k 개만 되살리고 나머지는 그대로 자른다.
            if int(survivors.sum()) < policy.top_k:
                cand = np.where(live, upper, -np.inf)
                keep = topk_indices(cand, min(policy.top_k, int(live.sum())))
                survivors = np.zeros_like(live)
                survivors[keep] = True
                kill = live & ~survivors

            if np.any(kill):
                frozen[kill] = s_m[kill]
                term_plane[kill] = m
                live = survivors
        live_history.append(live.copy())

    s_final = np.where(live, s_m, frozen)
    return StepResult(
        s_int=s_final,
        alive=live,
        term_plane=term_plane,
        read_live=read_live,
        theta_trace=theta_trace,
        live_count=live_count,
        n_planes=n_planes,
        n_active=n_tokens,
        extra={"theta_final": float(tracker.theta)},
    )


def masked_scores(result: StepResult, fill: float = -np.inf) -> np.ndarray:
    """종단된 토큰을 제외한 점수.

    하드웨어의 출력은 "상위 k 집합"이므로, 종단된 토큰은 어텐션에서 빠진다.
    정확도 평가는 이 마스킹된 점수로 softmax 를 재정규화해 수행한다.
    """
    out = result.s_int.astype(np.float64).copy()
    out[~result.alive] = fill
    return out
