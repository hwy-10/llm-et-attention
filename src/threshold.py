"""기준값 관리부 (θ) — 배경지식 가이드 5.4 / 6.2 / 6.3-(3)절

θ 는 **활성 토큰들의 하한 L_m 중 k번째로 큰 값**이다.

  하한이 θ 이상인 토큰이 최소 k개 있고, 그 토큰들의 실제 점수는
  각자의 하한 이상이므로 θ 이상이다.
  ⇒ 상한이 θ 보다 작은 토큰은 상위 k 에 들 수 없다.  (무손실 판정)

θ 를 언제 확정하느냐가 절감량을 좌우한다 (가이드 6.3-(3)):

    이르게 확정 -> θ 가 낮음 -> 종단 적음
    늦게 확정   -> θ 가 높음 -> 종단 많으나 이미 읽은 양이 많음

그래서 정책을 갈아끼울 수 있게 분리해 두었다. exp3 가 이 축을 스윕한다.

  every_plane  : 매 평면마다 갱신 (가이드 6.2절의 최종 형태)     [무손실]
  once_at_m    : 평면 m 개 처리 시점에 한 번 계산하고 고정        [무손실]
  prev_step    : 직전 디코드 스텝의 k번째 확정 점수를 그대로 사용  [★손실 발생★]
  oracle_fixed : 이번 스텝의 참 k번째 점수 (구현 불가, 참조선)     [무손실]

★★ 무손실성이 성립하는 조건 ★★

θ 가 **이번 스텝의 하한(또는 참값)** 에서 나와야 무손실이다. 그때는

    참 top-k 토큰 j 에 대해   upper_j ≥ s_j ≥ θ

가 보장되므로 절대 잘리지 않는다.

그런데 prev_step 은 θ 를 **직전 스텝**에서 가져온다. 직전 스텝의 k번째 점수가
이번 스텝의 k번째 점수보다 크면 참 top-k 가 잘려 나갈 수 있다.
즉 prev_step 은 부분 정렬 회로를 없애는 대신 무손실 보장을 포기하는 정책이다.
exp3 가 이 손실을 정량화한다 — "회로 단순화 vs 정확도"의 교환이며,
margin=0 인데도 손실이 나는 유일한 정책이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POLICIES = ("every_plane", "once_at_m", "prev_step", "oracle_fixed")
MARGIN_MODES = ("relative_gap", "relative_width", "absolute", "relative_qpos")


@dataclass
class ThetaPolicy:
    """θ 갱신 정책."""

    name: str = "every_plane"
    top_k: int = 8
    once_at_m: int = 3
    margin: float = 0.0
    margin_mode: str = "relative_gap"

    def __post_init__(self) -> None:
        if self.name not in POLICIES:
            raise ValueError(f"unknown theta policy {self.name!r}; choose from {POLICIES}")
        if self.margin_mode not in MARGIN_MODES:
            raise ValueError(f"unknown margin mode {self.margin_mode!r}")

    @property
    def is_exact(self) -> bool:
        """margin=0 이면 손실 없는 판정 (가이드 5.6절 정확 모드)."""
        return self.margin == 0.0

    def margin_abs(self, width: float, q_pos: float, gap: float = 0.0) -> float:
        """이번 평면에서 θ 에 더할 여유값 (절대 점수 단위).

        relative_gap  : ★ 기본값 ★  gap = (활성 토큰 하한의 최댓값 − θ) 에 비례.
                        즉 "1등과 k등 사이의 간격"을 단위로 삼는다.
                        margin=0   -> 정확 모드 (θ 그대로)
                        margin=1.0 -> θ 를 1등 수준까지 올림 (매우 공격적)
                        점수 분포의 실제 산포에 맞춰 정규화되므로 노브가 잘 듣는다.

        relative_width: 불확실성 폭(R_m − L_m offset)에 비례.
                        ⚠ 이 폭은 초반 평면에서 점수 산포보다 한 자릿수 크다.
                          그래서 margin 을 키워도 잘 듣지 않는다. 참고용으로만 둔다.
        relative_qpos : R_0 = 255·Q+ 에 비례 (평면과 무관한 고정 여유값).
        absolute      : 점수 단위 그대로.
        """
        if self.margin == 0.0:
            return 0.0
        if self.margin_mode == "relative_gap":
            return self.margin * max(float(gap), 0.0)
        if self.margin_mode == "relative_width":
            return self.margin * float(width)
        if self.margin_mode == "relative_qpos":
            return self.margin * 255.0 * float(q_pos)
        return float(self.margin)


def kth_largest(values: np.ndarray, live: np.ndarray, k: int) -> float:
    """활성 원소 중 k번째로 큰 값. 활성 개수가 k 미만이면 -inf."""
    n_live = int(live.sum())
    if n_live < k or k <= 0:
        return -np.inf
    masked = np.where(live, values, -np.inf)
    return float(np.partition(masked, -k)[-k])


class ThetaTracker:
    """한 디코드 스텝 동안 θ 를 관리한다."""

    def __init__(self, policy: ThetaPolicy, prev_theta: float | None = None,
                 oracle_theta: float | None = None):
        self.policy = policy
        self._theta = -np.inf
        self._frozen = False

        if policy.name == "prev_step":
            self._theta = -np.inf if prev_theta is None else float(prev_theta)
            self._frozen = True
        elif policy.name == "oracle_fixed":
            self._theta = -np.inf if oracle_theta is None else float(oracle_theta)
            self._frozen = True

    @property
    def theta(self) -> float:
        return self._theta

    def update(self, lower: np.ndarray, live: np.ndarray, m: int) -> float:
        """평면 m 개를 처리한 직후 호출. 갱신된 θ 를 반환한다."""
        p = self.policy
        if self._frozen:
            return self._theta

        if p.name == "every_plane":
            self._theta = kth_largest(lower, live, p.top_k)
        elif p.name == "once_at_m":
            if m == p.once_at_m:
                self._theta = kth_largest(lower, live, p.top_k)
                self._frozen = True
        return self._theta


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """상위 k 인덱스 (점수 내림차순).

    -inf 로 마스킹된 자리는 뽑지 않는다. 종단된 토큰은 전부 -inf 로 동률이라
    임의로 순서가 정해지는데, 그것이 top-k 에 섞이면 보존율이 부풀려진다.
    살아남은 것이 k 개보다 적으면 그만큼만 돌려준다.
    """
    scores = np.asarray(scores)
    k = int(min(k, scores.shape[-1]))
    if k <= 0:
        return np.empty(0, dtype=np.int64)

    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size == 0:
        return np.empty(0, dtype=np.int64)

    k = int(min(k, finite.size))
    idx = finite[np.argpartition(-scores[finite], k - 1)[:k]]
    return idx[np.argsort(-scores[idx])]
