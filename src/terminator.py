"""조기 종단 제어기 + 앞단 모듈 통합 인터페이스.

이 파일의 핵심 역할
------------------
앞단(masked_sum)이 평면별 부분내적 P_b를 만들면, 이 모듈이 그것을 한 평면씩 받아

    P_b -> S_m 누산 -> Q+/Q- 기반 L_m/U_m -> theta -> kill/live

순서로 종단 여부를 결정한다.

기존 `run_step()` API는 그대로 유지하므로 `designs.py`, `decode_loop.py`,
experiments 코드를 바꿀 필요가 없다.

★ 2026-08-29 채택 경위 ★
    팀원이 만든 v2(`terminator_v2`) 를 검증 후 채택했다. 기존 `run_step` 구현을 감싸는 형태라
    **동작이 완전히 같다** — 무작위 5,000회 + 경계 64조합 + 극단 12조합에서
    s_int / alive / term_plane / read_live / live_count / theta_trace 전부 일치.
    비용은 run_decode 기준 1.13~1.17배 (평면마다 PlaneDecision 을 만든다).

    얻은 것 세 가지
      1. 평면별 관측 — RTL 파형과 대조하려면 중간값이 필요하다 (ARCHITECTURE 10절)
      2. run_step_from_frontend — 팀1 -> 팀2 인계를 한 함수로
      3. 평면수 불일치 차단 — 예전 구현은 partials 가 **더 적을 때** 예외 없이
         통과해 조용히 틀린 답을 냈다

    채택하면서 고친 것
      run_step_from_frontend 가 top_k=8 / once_at_m=3 / margin_mode="relative_gap"
      을 자체 기본값으로 들고 있었다. 셋 다 우리가 이미 고친 값의 **옛 버전**이고,
      특히 relative_gap 은 relative_width 확정을 우회해 되돌리고 있었다.
      -> None 이면 ThetaPolicy 기본값을 따르게 바꿨다.

추가된 핵심 API
---------------
1) TerminationController
   - 실제 하드웨어처럼 평면 하나씩 `process_plane()`에 넣는 스트리밍 인터페이스.

2) run_step_from_frontend()
   - 앞단이 준 `q_stored`와 `partials`를 바로 받아
     Q+/Q-를 스텝 시작 시 딱 한 번 계산한 뒤 종단까지 실행하는 통합 함수.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bounds import StepBounds, step_bounds
from .quantize import N_PLANES, plane_weights
from .threshold import ThetaPolicy, ThetaTracker, topk_indices


@dataclass
class StepResult:
    """디코드 스텝 하나의 최종 결과."""

    s_int: np.ndarray
    alive: np.ndarray
    term_plane: np.ndarray
    read_live: np.ndarray
    theta_trace: np.ndarray
    live_count: np.ndarray
    n_planes: int = N_PLANES
    n_active: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def mean_term_plane(self) -> float:
        return (
            float(self.term_plane.mean())
            if self.term_plane.size
            else float(self.n_planes)
        )

    @property
    def survivor_frac(self) -> float:
        return float(self.alive.mean()) if self.alive.size else 1.0


@dataclass(frozen=True)
class PlaneDecision:
    """평면 하나를 처리한 직후의 종단 판정 결과.

    앞단/RTL과 인터페이스를 맞출 때 보기 위한 구조체다.
    """

    plane_index: int
    m: int
    read_mask: np.ndarray
    live_before: np.ndarray
    live_after: np.ndarray
    killed: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    theta: float
    margin_abs: float


class TerminationController:
    """Q+/Q- 레지스터 + 종단 판정을 스트리밍 방식으로 통합한 제어기.

    Parameters
    ----------
    bounds:
        현재 decode step의 Q+/Q-. 스텝 시작 시 한 번만 계산/로드된다.
    policy:
        theta 정책.
    n_tokens:
        이번 step에서 causal mask 안에 들어오는 활성 key token 수.
    decision_latency:
        종단 판정이 실제 memory read mask에 반영될 때까지의 plane 지연.
    """

    def __init__(
        self,
        bounds: StepBounds,
        policy: ThetaPolicy,
        n_tokens: int,
        *,
        decision_latency: int = 0,
        prev_theta: float | None = None,
        oracle_theta: float | None = None,
        enable_termination: bool = True,
    ) -> None:
        n_tokens = int(n_tokens)
        decision_latency = int(decision_latency)

        if n_tokens < 0:
            raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")
        if decision_latency < 0:
            raise ValueError(
                f"decision_latency must be >= 0, got {decision_latency}"
            )
        if bounds.n_planes <= 0:
            raise ValueError(
                f"bounds.n_planes must be > 0, got {bounds.n_planes}"
            )

        self.bounds = bounds
        self.policy = policy
        self.n_tokens = n_tokens
        self.n_planes = int(bounds.n_planes)
        self.decision_latency = decision_latency
        self.enable_termination = bool(enable_termination)

        self.weights = plane_weights(self.n_planes)
        self.tracker = ThetaTracker(
            policy,
            prev_theta=prev_theta,
            oracle_theta=oracle_theta,
        )

        # ---- step 시작 시 초기화되는 레지스터/상태 ----------------------
        self.s_m = np.zeros(n_tokens, dtype=np.int64)
        self.frozen = np.zeros(n_tokens, dtype=np.int64)
        self.live = np.ones(n_tokens, dtype=bool)
        self.term_plane = np.full(
            n_tokens, self.n_planes, dtype=np.int32
        )

        # live_history[m] = m개 평면 처리 후 논리적 live 상태
        self.live_history: list[np.ndarray] = [self.live.copy()]

        self.read_live = np.zeros(
            (self.n_planes, n_tokens), dtype=bool
        )
        self.theta_trace = np.full(
            self.n_planes, -np.inf, dtype=np.float64
        )
        self.live_count = np.zeros(
            self.n_planes, dtype=np.int64
        )

        self._next_plane = 0

    @property
    def done(self) -> bool:
        return self._next_plane >= self.n_planes

    @property
    def next_plane(self) -> int:
        return self._next_plane

    def process_plane(self, partial: np.ndarray) -> PlaneDecision:
        """앞단에서 계산된 P_b 한 평면을 받아 종단 판정까지 수행한다.

        `partial[j] = P_b(j)` 이며, 호출 순서는 반드시 MSB -> LSB다.
        """

        if self.done:
            raise RuntimeError("all bit-planes have already been processed")

        t = self._next_plane
        m = t + 1

        p = np.asarray(partial, dtype=np.int64).reshape(-1)
        if p.shape != (self.n_tokens,):
            raise ValueError(
                f"partial for plane {t} must have shape "
                f"({self.n_tokens},), got {p.shape}"
            )

        live_before = self.live.copy()

        # ---------------------------------------------------------------
        # 1. 실제 memory read에 적용되는 mask
        #    판정 latency 때문에 논리적 live보다 늦게 반영될 수 있다.
        # ---------------------------------------------------------------
        mask_idx = max(0, t - self.decision_latency)
        read_mask = self.live_history[mask_idx].copy()
        self.read_live[t] = read_mask
        self.live_count[t] = int(live_before.sum())

        # ---------------------------------------------------------------
        # 2. 앞단 P_b를 현재 부분점수 S_m에 통합
        #    이미 죽은 token은 accumulator를 동결한다.
        # ---------------------------------------------------------------
        self.s_m = (
            self.s_m
            + self.weights[t] * p * live_before.astype(np.int64)
        )

        # 현재 평면까지 읽었을 때의 bracket.
        lower = self.s_m + self.bounds.l_offset(m)
        upper = self.s_m + self.bounds.r(m)

        killed = np.zeros(self.n_tokens, dtype=bool)
        margin_abs = 0.0

        # 마지막 평면에서는 이미 정확한 점수이므로 더 죽일 필요가 없다.
        if self.enable_termination and m < self.n_planes:
            theta = self.tracker.update(lower, live_before, m)

            if np.isfinite(theta):
                if np.any(live_before):
                    best_lower = float(
                        np.max(
                            np.where(
                                live_before,
                                lower,
                                -np.inf,
                            )
                        )
                    )
                    gap = best_lower - theta
                else:
                    gap = 0.0

                margin_abs = self.policy.margin_abs(
                    self.bounds.width(m),
                    self.bounds.q_pos,
                    gap,
                )

                # 핵심 판정:
                # U_m < theta + margin 이면 top-k 후보가 될 수 없으므로 kill.
                killed = (
                    live_before
                    & (upper < theta + margin_abs)
                )
                survivors = live_before & ~killed

                # 근사 모드에서도 최소 top_k개는 남긴다.
                # 전부 pruning을 취소하지 않고 upper가 큰 token만 복구한다.
                if int(survivors.sum()) < self.policy.top_k:
                    cand = np.where(
                        live_before,
                        upper,
                        -np.inf,
                    )
                    keep = topk_indices(
                        cand,
                        min(
                            self.policy.top_k,
                            int(live_before.sum()),
                        ),
                    )
                    survivors = np.zeros_like(live_before)
                    survivors[keep] = True
                    killed = live_before & ~survivors

                if np.any(killed):
                    self.frozen[killed] = self.s_m[killed]
                    self.term_plane[killed] = m
                    self.live = survivors
        else:
            theta = self.tracker.theta

        self.theta_trace[t] = theta
        self.live_history.append(self.live.copy())
        self._next_plane += 1

        return PlaneDecision(
            plane_index=t,
            m=m,
            read_mask=read_mask,
            live_before=live_before,
            live_after=self.live.copy(),
            killed=killed.copy(),
            lower=lower.copy(),
            upper=upper.copy(),
            theta=float(theta),
            margin_abs=float(margin_abs),
        )

    def finish(self) -> StepResult:
        """8개 평면 처리가 끝난 뒤 최종 결과를 반환한다."""

        if not self.done:
            raise RuntimeError(
                f"step is not finished: processed "
                f"{self._next_plane}/{self.n_planes} planes"
            )

        s_final = np.where(
            self.live,
            self.s_m,
            self.frozen,
        )

        return StepResult(
            s_int=s_final,
            alive=self.live.copy(),
            term_plane=self.term_plane.copy(),
            read_live=self.read_live.copy(),
            theta_trace=self.theta_trace.copy(),
            live_count=self.live_count.copy(),
            n_planes=self.n_planes,
            n_active=self.n_tokens,
            extra={
                "theta_final": float(self.tracker.theta),
                "q_pos": int(self.bounds.q_pos),
                "q_neg": int(self.bounds.q_neg),
            },
        )


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
    """기존 배치 API.

    기존 `designs.py`/`decode_loop.py`와의 호환성을 유지하기 위한 wrapper다.
    내부에서는 `TerminationController`에 MSB부터 P_b를 한 평면씩 넣는다.
    """

    p = np.asarray(partials, dtype=np.int64)
    if p.ndim != 2:
        raise ValueError(
            f"partials must be (n_planes, n_tokens), got {p.shape}"
        )

    n_planes, n_tokens = p.shape

    if int(bounds.n_planes) != n_planes:
        raise ValueError(
            "plane-count mismatch between front-end partials and bounds: "
            f"partials={n_planes}, bounds={bounds.n_planes}"
        )

    controller = TerminationController(
        bounds,
        policy,
        n_tokens,
        decision_latency=decision_latency,
        prev_theta=prev_theta,
        oracle_theta=oracle_theta,
        enable_termination=enable_termination,
    )

    for t in range(n_planes):
        controller.process_plane(p[t])

    return controller.finish()


def run_step_from_frontend(
    q_stored: np.ndarray,
    partials: np.ndarray,
    *,
    top_k: int | None = None,
    theta_policy: str | None = None,
    once_at_m: int | None = None,
    margin: float | None = None,
    margin_mode: str | None = None,
    decision_latency: int = 0,
    prev_theta: float | None = None,
    oracle_theta: float | None = None,
    enable_termination: bool = True,
) -> StepResult:
    """★ 앞 조 -> 우리 조를 직접 잇는 통합 함수.

    Inputs
    ------
    q_stored:
        앞단 query 양자화 결과. 이 값으로 Q+/Q-를 **딱 한 번** 계산한다.
    partials:
        앞단 masked_sum이 만든 P_b.
        shape = (n_planes, n_active), index 0 = MSB.

    Flow
    ----
        q_stored --step_bounds--> Q+/Q- register
                                      |
        partials ---------------------+--> run_step
                                           |
                                           +--> S_m
                                           +--> L_m/U_m
                                           +--> theta
                                           +--> kill/live
    """

    p = np.asarray(partials, dtype=np.int64)
    if p.ndim != 2:
        raise ValueError(
            f"partials must be (n_planes, n_tokens), got {p.shape}"
        )

    n_planes = int(p.shape[0])

    # ★ 스텝 시작 시 한 번만 계산.
    bounds = step_bounds(q_stored, n_planes=n_planes)

    # ★ 기본값을 여기 적지 않는다 ★
    #   편의 함수가 자체 기본값을 들고 있으면 설정이 조용히 무시된다.
    #   실제로 초판은 top_k=8 / once_at_m=3 / margin_mode="relative_gap" 을 들고
    #   있었는데, 셋 다 우리가 이미 고친 값의 **옛 버전**이었다. 특히
    #   relative_gap 은 relative_width 로 확정하고 ThetaPolicy 기본값까지 바꾼
    #   것을 이 함수가 우회해 되돌리고 있었다.
    #   -> None 이면 ThetaPolicy 의 기본값(= 확정값)을 따른다.
    d = ThetaPolicy()
    policy = ThetaPolicy(
        name=d.name if theta_policy is None else theta_policy,
        top_k=d.top_k if top_k is None else int(top_k),
        once_at_m=d.once_at_m if once_at_m is None else int(once_at_m),
        margin=d.margin if margin is None else float(margin),
        margin_mode=d.margin_mode if margin_mode is None else margin_mode,
    )

    return run_step(
        p,
        bounds,
        policy,
        decision_latency=decision_latency,
        prev_theta=prev_theta,
        oracle_theta=oracle_theta,
        enable_termination=enable_termination,
    )


def masked_scores(
    result: StepResult,
    fill: float = -np.inf,
) -> np.ndarray:
    """종단된 token을 attention 후보에서 제외한다."""

    out = result.s_int.astype(np.float64).copy()
    out[~result.alive] = fill
    return out

