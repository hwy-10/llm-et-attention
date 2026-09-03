"""잔여 상한 R_m 과 하한 L_m  — 배경지식 가이드 5.4절

    Q+ = q 원소 중 양수의 합        (q 에만 의존 → 디코드 스텝당 1회 계산)
    Q− = q 원소 중 음수의 합        (0 이하)

    상위 m 개 평면 처리 후:
        S_m = Σ_{처리한 평면} 2^b · P_b        (확정된 부분 점수)
        R_m = (2^(8−m) − 1) · Q+              (남은 평면의 최대 기여)
        L_m = S_m + (2^(8−m) − 1) · Q−        (하한)

        ⇒  L_m  ≤  s_j  ≤  S_m + R_m

이 부등식이 항상 성립한다는 것이 조기 종단의 무손실성을 보장한다.
tests/test_bounds.py 가 랜덤 케이스로 이를 검증한다.

★★ 성립 조건 — 2026-08-28 반례 탐색에서 확인 ★★

    0 <= K_stored < 2^n_planes

  이 범위를 벗어나면 to_bitplanes() 가 uint16 으로 감싸며 **조용히 자르므로**
  비트평면이 K 를 대표하지 못하고, 그 순간 상한식이 깨진다.

      K = 256 -> 복원값 0        K = -1 -> 복원값 255

  quantize_key() 는 항상 0..255 를 내므로 정상 경로에서는 걸리지 않는다. 그러나
  to_bitplanes() 는 공개 함수이고 decode_loop 가 직접 부른다. n_planes 를 8보다
  작게 두고 K 를 0..255 로 넣는 경우도 같은 이유로 깨진다(표현 범위 초과).
  verify_bracket() 도 이 경우를 못 잡는다 — S_m 과 참값이 둘 다 오염되어
  서로는 일관되기 때문이다.

★ 하드웨어 대응 ★
  Q+ / Q− 는 스텝 시작 시 한 번 계산해 레지스터에 넣어 두고,
  매 평면마다 시프트 한 번((2^(8−m)−1) 곱)으로 R_m, L_m 을 얻는다.
  (가이드 6.2절 "조기 종단 제어기")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quantize import N_PLANES, remaining_scale


@dataclass(frozen=True)
class StepBounds:
    """한 디코드 스텝(= 하나의 q 벡터)에 대한 상/하한 상수.

    q 는 스텝 안에서 모든 토큰에 공통이므로, 이 객체도 스텝당 하나면 된다.
    """

    q_pos: int          # Q+
    q_neg: int          # Q−  (0 이하)
    n_planes: int = N_PLANES

    # 평면 m 개 처리 후의 잔여 계수 (2^(n−m) − 1), m = 0..n
    @property
    def coeffs(self) -> np.ndarray:
        return np.array(
            [remaining_scale(m, self.n_planes) for m in range(self.n_planes + 1)],
            dtype=np.int64,
        )

    def r(self, m: int) -> int:
        """R_m — 남은 평면의 최대 기여."""
        return int(remaining_scale(m, self.n_planes)) * self.q_pos

    def l_offset(self, m: int) -> int:
        """L_m − S_m — 남은 평면의 최소 기여 (0 이하)."""
        return int(remaining_scale(m, self.n_planes)) * self.q_neg

    def width(self, m: int) -> int:
        """불확실성 폭 (상한 − 하한). 근사 마진의 정규화 기준."""
        return self.r(m) - self.l_offset(m)

    def upper(self, s_m: np.ndarray, m: int) -> np.ndarray:
        return s_m + self.r(m)

    def lower(self, s_m: np.ndarray, m: int) -> np.ndarray:
        return s_m + self.l_offset(m)


def step_bounds(q_stored: np.ndarray, n_planes: int = N_PLANES) -> StepBounds:
    """q 벡터 하나로부터 Q+ / Q− 를 계산.

    Parameters
    ----------
    q_stored : (head_dim,) signed 정수 (quantize_query 결과)
    """
    q = np.asarray(q_stored, dtype=np.int64).ravel()
    return StepBounds(
        q_pos=int(q[q > 0].sum()),
        q_neg=int(q[q < 0].sum()),
        n_planes=n_planes,
    )


def batch_step_bounds(q_stored: np.ndarray, n_planes: int = N_PLANES) -> tuple[np.ndarray, np.ndarray]:
    """여러 스텝의 Q+ / Q− 를 한 번에. (n_steps, head_dim) -> 두 개의 (n_steps,)"""
    q = np.asarray(q_stored, dtype=np.int64)
    return (
        np.where(q > 0, q, 0).sum(axis=-1),
        np.where(q < 0, q, 0).sum(axis=-1),
    )


def max_possible_score(q_pos: int, n_planes: int = N_PLANES) -> int:
    """s_j 가 가질 수 있는 이론적 최대 (= R_0). 마진/오차 정규화에 쓴다."""
    return int(remaining_scale(0, n_planes)) * int(q_pos)


def verify_bracket(
    s_m: np.ndarray, true_scores: np.ndarray, bounds: StepBounds, m: int, atol: int = 0
) -> bool:
    """L_m ≤ s ≤ S_m + R_m 이 전부 성립하는지 확인 (테스트/디버그용)."""
    lo = bounds.lower(s_m, m)
    hi = bounds.upper(s_m, m)
    return bool(np.all(true_scores >= lo - atol) and np.all(true_scores <= hi + atol))
