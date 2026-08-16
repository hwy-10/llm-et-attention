"""양자화 + 비트평면 분해  — 배경지식 가이드 5.2 / 5.3절

★★ 이 파일의 규약이 5.4절 상한식(R_m = (2^(8-m)-1)·Q+)의 전제다 ★★

  K : 비대칭 unsigned 8비트.  K_stored ∈ [0, 255]
      실제값 ≈ (K_stored − z) · scale_k
      비트 자리값이 전부 양수(2^b)이므로 "남은 비트가 전부 1"이 곧 최대 기여가 된다.
      ▶ signed(2의 보수)로 저장하면 MSB 자리값이 −128이 되어 상한식이 깨진다.

  q : 대칭 signed 8비트.  q ∈ [−127, 127]

  zero-point 보정: s_real = scale_q·scale_k · ( q·K_stored − z·Σq )
      두 번째 항은 모든 토큰 j 에 동일한 상수이므로
        * 순위(top-k) 판정에는 영향 없음  → 종단 로직은 q·K_stored 만 본다
        * 최종 점수값에는 필요           → accumulator.py 가 더한다
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_PLANES = 8


# ---------------------------------------------------------------------------
# Key — 비대칭 unsigned
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KeyQuant:
    """K 양자화 결과.  실제값 ≈ (stored − zero_point) * scale"""

    stored: np.ndarray       # (n_tokens, head_dim) uint8, 값 [0, 255]
    scale: np.ndarray        # per_channel 이면 (1, head_dim), 아니면 스칼라
    zero_point: np.ndarray   # 같은 shape, 정수값

    @property
    def n_tokens(self) -> int:
        return self.stored.shape[0]

    @property
    def head_dim(self) -> int:
        return self.stored.shape[1]

    def dequantize(self) -> np.ndarray:
        return (self.stored.astype(np.float64) - self.zero_point) * self.scale


def quantize_key(
    k: np.ndarray,
    bits: int = 8,
    granularity: str = "per_channel",
    clip_percentile: float = 100.0,
) -> KeyQuant:
    """K 를 비대칭 unsigned 정수로 양자화.

    granularity
      per_channel : 차원(열)마다 별도 scale/zero-point — KIVI[11] 권장
      per_tensor  : 전체 하나
    """
    k = np.asarray(k, dtype=np.float64)
    if k.ndim != 2:
        raise ValueError(f"k must be (n_tokens, head_dim), got {k.shape}")
    qmax = (1 << bits) - 1

    axis = 0 if granularity == "per_channel" else None
    if clip_percentile >= 100.0:
        lo = np.min(k, axis=axis, keepdims=True)
        hi = np.max(k, axis=axis, keepdims=True)
    else:
        p = clip_percentile
        lo = np.percentile(k, 100.0 - p, axis=axis, keepdims=True)
        hi = np.percentile(k, p, axis=axis, keepdims=True)

    scale = np.maximum(hi - lo, 1e-12) / qmax
    zero_point = np.rint(-lo / scale)
    stored = np.clip(np.rint(k / scale) + zero_point, 0, qmax).astype(np.uint8)
    return KeyQuant(stored=stored, scale=np.asarray(scale), zero_point=np.asarray(zero_point))


# ---------------------------------------------------------------------------
# Query — 대칭 signed
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QueryQuant:
    stored: np.ndarray       # (..., head_dim) int16 담김, 값 [-127, 127]
    scale: np.ndarray

    def dequantize(self) -> np.ndarray:
        return self.stored.astype(np.float64) * self.scale


def quantize_query(q: np.ndarray, bits: int = 8, granularity: str = "per_token") -> QueryQuant:
    q = np.asarray(q, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1
    axis = -1 if granularity == "per_token" else None
    amax = np.max(np.abs(q), axis=axis, keepdims=True)
    scale = np.maximum(amax, 1e-12) / qmax
    stored = np.clip(np.rint(q / scale), -qmax, qmax).astype(np.int16)
    return QueryQuant(stored=stored, scale=np.asarray(scale))


# ---------------------------------------------------------------------------
# 비트평면
# ---------------------------------------------------------------------------
def to_bitplanes(stored: np.ndarray, n_planes: int = N_PLANES) -> np.ndarray:
    """unsigned 정수 -> 비트평면.

    (n_tokens, head_dim) -> (n_planes, n_tokens, head_dim) uint8
    ★ 첫 축 index 0 = MSB (b7), index 7 = LSB (b0) — MSB 우선 순서로 저장한다.
      이렇게 두면 "평면 m 만 읽기"가 곧 배열의 앞쪽 슬라이스가 된다.
    """
    u = np.asarray(stored, dtype=np.uint16)
    shifts = np.arange(n_planes - 1, -1, -1, dtype=np.uint16)  # MSB first
    return ((u[None, ...] >> shifts[:, None, None]) & 1).astype(np.uint8)


def plane_weights(n_planes: int = N_PLANES) -> np.ndarray:
    """MSB 우선 순서에 대응하는 자리값. [128, 64, ..., 1] — 전부 양수."""
    return (1 << np.arange(n_planes - 1, -1, -1, dtype=np.int64)).astype(np.int64)


def from_bitplanes(planes: np.ndarray) -> np.ndarray:
    """to_bitplanes 의 역변환. 무손실이어야 한다."""
    w = plane_weights(planes.shape[0])
    return np.tensordot(w, planes.astype(np.int64), axes=(0, 0))


def remaining_scale(m: int, n_planes: int = N_PLANES) -> int:
    """상위 m 개 평면을 처리한 뒤, 남은 평면들의 자리값 합.

        Σ_{b=0}^{n_planes-1-m} 2^b  =  2^(n_planes-m) − 1

    가이드 5.4절의 (2^(8-m) − 1) 계수.
    """
    return (1 << (n_planes - m)) - 1
