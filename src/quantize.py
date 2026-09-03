"""양자화 + 비트평면 분해  — 배경지식 가이드 5.2 / 5.3절

★★ 이 파일의 규약이 5.4절 상한식(R_m = (2^(8-m)-1)·Q+)의 전제다 ★★

  K : 비대칭 unsigned 8비트.  K_stored ∈ [0, 255]
      실제값 ≈ (K_stored − z) · scale_k
      비트 자리값이 전부 양수(2^b)이므로 "남은 비트가 전부 1"이 곧 최대 기여가 되고,
      8개 평면 전부에 R_m 공식이 단일 형태로 적용된다.

      ★ 정정 (2026-08) ★
      "signed 로 저장하면 상한식이 깨진다"는 이전 서술은 틀렸다. MSB-first 이므로
      부호 비트는 라운드 0에 확정되어 항상 '결정된 부분합'에 들어가고, 미확정으로
      남는 비트의 자리값은 전부 양수다. 따라서 signed 에서도 유효한 bound 를 세울 수
      있고, PADE(HPCA 2026)/BitStopper 가 2의 보수에서 그렇게 유도한다.
      ★ 단, 정정 문구가 빠뜨린 조건이 있다 (2026-08-28 직접 유도해 확인) ★
        m = 0 (아무 평면도 안 읽은 시점) 에서는 signed 가 **다른 공식**을 쓴다.
              R_0 = -128*Q- + 127*Q+       (unsigned 는 255*Q+)
              unsigned 공식을 그대로 쓰면 Q+ < |Q-| 인 q 에서 상한을 밑돌아 깨진다.
              반례: q = [1]x10 + [-100]x54  ->  참 최대 692,470 vs 255*Q+ = 2,550
        m >= 1 에서는 남은 자리값이 [64,32,...,1] 로 전부 양수라 같은 형태가 된다.

      즉 unsigned 의 이득은 **평면 0의 예외 처리를 없애 8개 평면에 단일 공식을
      쓰는 것**이다. 회로 단순화이지 수학적 필연이 아니라는 결론은 그대로다.
      자세한 것은 docs/related_work.md 참조.

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

    # NaN / inf 는 clip 을 그대로 통과해 stored 가 전부 0 이 된다.
    # 범위 검사(0..255)는 멀쩡해 보이는데 값만 사라지므로 여기서 막는다.
    if not np.all(np.isfinite(k)):
        n_bad = int((~np.isfinite(k)).sum())
        raise ValueError(f"k has {n_bad} non-finite value(s); NaN/inf quantize to 0 silently")

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

    (...) -> (n_planes, ...) uint8.  입력 차원은 몇 이어도 된다.
    ★ 첫 축 index 0 = MSB (b7), index -1 = LSB (b0) — MSB 우선 순서로 저장한다.
      이렇게 두면 "평면 m 만 읽기"가 곧 배열의 앞쪽 슬라이스가 된다.
    """
    u = np.asarray(stored)

    # 실수를 받으면 예전에는 uint16 캐스트가 소수점을 조용히 버렸다 (3.7 -> 3)
    if not np.issubdtype(u.dtype, np.integer):
        raise ValueError(f"stored must be an integer array, got dtype {u.dtype}")

    # 담기지 않는 값은 조용히 잘린다 (256 -> 0, -1 -> 255).
    # ★ 그러면 비트평면이 K 를 대표하지 못하고 5.4절 상한식이 깨진다
    #   (반례 탐색에서 위반 4~9건 확인). -> src/bounds.py 독스트링
    # 왕복이 무손실인 구간은 0 .. 2^n_planes-1 뿐이므로 벗어나면 막는다.
    lo, hi = int(u.min()) if u.size else 0, int(u.max()) if u.size else 0
    limit = (1 << n_planes) - 1
    if lo < 0 or hi > limit:
        raise ValueError(
            f"stored must be in [0, {limit}] for {n_planes} planes, got [{lo}, {hi}]"
        )

    # uint16 으로 캐스트하지 않는다. 위 가드는 2^n_planes-1 까지 허용하는데
    # uint16 은 65535 에서 잘려, n_planes > 16 이면 가드를 통과한 값이 사라졌다.
    shifts = np.arange(n_planes - 1, -1, -1, dtype=np.int64)   # MSB first
    shifts = shifts.reshape((n_planes,) + (1,) * u.ndim)       # 차원에 맞춰 편다
    return ((u[None, ...] >> shifts) & 1).astype(np.uint8)


def plane_weights(n_planes: int = N_PLANES) -> np.ndarray:
    """MSB 우선 순서에 대응하는 자리값. [128, 64, ..., 1] — 전부 양수."""
    return 1 << np.arange(n_planes - 1, -1, -1, dtype=np.int64)


def from_bitplanes(planes: np.ndarray, dtype=np.int64) -> np.ndarray:
    """to_bitplanes 의 역변환. 무손실이어야 한다.

    dtype 기본값이 int64 인 것은 취향이 아니다. 좁은 타입을 기본으로 두면
    n_planes > 8 에서 예외 없이 값이 잘린다 (4095 -> 255). 아래쪽 경로가
    전부 int64 정수 연산이라 여기만 좁으면 그 사슬이 끊긴다.
    """
    w = plane_weights(planes.shape[0])
    return np.tensordot(w, planes.astype(np.int64), axes=(0, 0)).astype(dtype)


def remaining_scale(m: int, n_planes: int = N_PLANES) -> int:
    """상위 m 개 평면을 처리한 뒤, 남은 평면들의 자리값 합.

        Σ_{b=0}^{n_planes-1-m} 2^b  =  2^(n_planes-m) − 1

    가이드 5.4절의 (2^(8-m) − 1) 계수.
    """
    # m=-1 은 511(평면 9개분)을 조용히 내고, m=9 는 시프트 오류만 낸다
    if not 0 <= m <= n_planes:
        raise ValueError(f"m = {m}: must be in [0, {n_planes}] (planes processed so far)")
    return (1 << (n_planes - m)) - 1
