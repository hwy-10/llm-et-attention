"""부분 내적 유닛 (Masked-Sum Unit) — 배경지식 가이드 5.3 / 6.2절

    P_b(j) = Σ_i q_i · K_j^(b)_i
           = ( K 의 b번째 비트가 1인 위치의 q 값들의 합 )

K 의 비트가 0/1 이므로 곱셈이 필요 없다. 마스킹 + 가산 트리만으로 계산된다.
→ FPGA 에서 DSP 를 쓰지 않고 LUT 만으로 구현 가능 (가이드 6.3-(1)의 전제).

이 파일은 두 가지를 제공한다.
  1) 수치 계산: 디코드 루프가 쓸 P_b 텐서
  2) 하드웨어 모델: 가산 트리의 폭·깊이·비트폭 → 자원 추정의 근거
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .quantize import N_PLANES


# ---------------------------------------------------------------------------
# 수치 계산
# ---------------------------------------------------------------------------
def _masked_sum(q: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Key 비트가 1인 위치의 q 값을 선택한 뒤 head_dim 방향으로 합산한다."""
    out = np.empty(
        (kp.shape[0], q.shape[0], kp.shape[1]),
        dtype=np.int32,
    )

    for plane in range(kp.shape[0]):
        for step in range(q.shape[0]):
            out[plane, step] = np.where(
                kp[plane] == 1,
                q[step],
                0,
            ).sum(axis=-1, dtype=np.int32)

    return out


def partial_dots(
    q_stored: np.ndarray,
    k_planes: np.ndarray,
    chunk: int = 0,
) -> np.ndarray:
    """모든 (평면, 스텝, 토큰) 조합의 부분 내적을 한 번에 계산한다.

    Parameters
    ----------
    q_stored : (n_steps, head_dim) 또는 (head_dim,) signed 정수
    k_planes : (n_planes, n_tokens, head_dim) uint8 — MSB 우선 순서
    chunk    : 토큰 축 청크 크기 (0 = 통짜). 큰 T 에서 메모리 제어용.

    Returns
    -------
    (n_planes, n_steps, n_tokens) int32
    첫 축 index 0 = MSB 평면.

    ★ 디코드 루프는 이 텐서를 한 번만 만들고, margin/θ정책/top-k 스윕에
      전부 재사용한다. 그래서 스윕이 빠르다.
    """
    q = np.atleast_2d(np.asarray(q_stored, dtype=np.int32))
    kp = np.asarray(k_planes)
    if kp.ndim != 3:
        raise ValueError(f"k_planes must be (n_planes, n_tokens, head_dim), got {kp.shape}")
    if q.shape[-1] != kp.shape[-1]:
        raise ValueError(f"head_dim mismatch: q={q.shape[-1]}, k_planes={kp.shape[-1]}")

    if not chunk or chunk >= kp.shape[1]:
        return _masked_sum(q, kp)

    outs = [
        _masked_sum(q, kp[:, start : start + chunk, :])
        for start in range(0, kp.shape[1], chunk)
    ]
    return np.concatenate(outs, axis=-1)


def partial_dot_single(q_stored: np.ndarray, k_plane: np.ndarray) -> np.ndarray:
    """평면 하나에 대한 P_b. (n_tokens, head_dim) x (head_dim,) -> (n_tokens,)"""
    return np.asarray(k_plane, dtype=np.int32) @ np.asarray(q_stored, dtype=np.int32)


# ---------------------------------------------------------------------------
# 하드웨어 모델 — 자원 추정의 근거 (가이드 6.3-(1))
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdderTreeModel:
    """d 입력 가산 트리 하나의 형상.

    기준 설계(①)는 INT8 곱셈기를 쓰므로 DSP 를 소모하고,
    제안 설계는 이 가산 트리만 쓰므로 LUT 를 소모한다.
    두 자원의 교환 관계가 유리한지는 합성 결과로 확인해야 한다.
    """

    n_inputs: int = 64          # = head_dim
    input_bits: int = 8         # q 의 비트폭 (signed)
    lut_per_fa: float = 1.0     # full-adder 1비트당 LUT (6-input LUT 기준 근사)

    @property
    def depth(self) -> int:
        """트리 깊이 = 임계 경로 단수."""
        return int(math.ceil(math.log2(max(self.n_inputs, 2))))

    @property
    def n_adders(self) -> int:
        return self.n_inputs - 1

    @property
    def output_bits(self) -> int:
        """부호 있는 d개 합의 비트폭."""
        return self.input_bits + self.depth

    @property
    def fully_pipelined_latency_cycles(self) -> int:
        """가산 트리의 모든 단계가 파이프라인된 경우의 예상 지연 시간."""
        return self.depth

    @property
    def est_lut(self) -> float:
        """가산 트리 하나의 LUT 추정.

        단수 k 에서 가산기 폭이 input_bits + k 이므로 총 비트 수를 합산한다.
        어디까지나 1차 추정이며, 논문에는 Vivado 실측을 쓴다.
        """
        total_bits = 0
        n, w = self.n_inputs, self.input_bits
        while n > 1:
            pairs = n // 2
            total_bits += pairs * (w + 1)
            n = (n + 1) // 2
            w += 1
        return total_bits * self.lut_per_fa

    @property
    def est_mask_lut(self) -> float:
        """비트마스킹 논리 (q_i AND K_bit) 의 LUT."""
        return self.n_inputs * self.input_bits / 2.0  # 6-LUT 하나가 2비트 처리 가정

    def summary(self) -> dict:
        return {
            "n_inputs": self.n_inputs,
            "depth": self.depth,
            "n_adders": self.n_adders,
            "output_bits": self.output_bits,
            "est_lut_tree": round(self.est_lut, 1),
            "est_lut_mask": round(self.est_mask_lut, 1),
            "est_lut_total": round(self.est_lut + self.est_mask_lut, 1),
            "uses_dsp": False,
        }


@dataclass(frozen=True)
class BaselineMacModel:
    """기준 설계(①): 병렬 INT8 곱셈 누산 구조."""

    n_pe: int = 32
    input_bits: int = 8
    dsp_per_mac: int = 1

    @property
    def est_dsp(self) -> int:
        return self.n_pe * self.dsp_per_mac

    def summary(self) -> dict:
        return {"n_pe": self.n_pe, "est_dsp": self.est_dsp, "uses_dsp": True}


def accumulator_bits(head_dim: int, q_bits: int = 8, n_planes: int = N_PLANES) -> int:
    """시프트 누산기의 비트폭.

    최대 |s| = (2^n_planes − 1) · (2^(q_bits−1) − 1) · head_dim
    """
    max_abs = ((1 << n_planes) - 1) * (1 << (q_bits - 1)) * head_dim
    return int(math.ceil(math.log2(max_abs + 1))) + 1  # +1 = 부호
