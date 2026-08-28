"""시프트 누산기 + zero-point 보정 — 배경지식 가이드 5.3 / 6.2절

    S_m = Σ_{처리한 평면 b} 2^b · P_b        (시프트는 배선, 비용 거의 없음)

★★ per-channel 양자화와 "곱셈기 없음"을 동시에 지키는 방법 ★★

K 를 차원별(per-channel) scale 로 양자화하면 (KIVI[11] 권장)

    s_real(j) = Σ_i (q_i·sq) · (K_st[j,i] − z_i)·sc_i

가 되어 sc_i 가 안쪽에 남는다. 이러면 곱셈이 되살아난다.

해결: **sc_i 를 q 에 미리 접어 넣는다.**

    q̃_i = q_real_i · sc_i           <- 소프트웨어에서 한 번 (스텝당 d회 곱셈)
    q̃ 를 대칭 양자화 -> q_st, sq

    s_real(j) = sq · ( Σ_i q_st_i · K_st[j,i]  −  Σ_i q_st_i · z_i )
                       └──── 마스크드 합 (곱셈 없음) ────┘   └─ 스텝당 상수 ─┘

두 번째 항은 모든 토큰 j 에 동일하다. 따라서
  * top-k 순위 판정에는 영향이 없다  -> terminator.py 는 정수 s_int 만 본다
  * 최종 점수값에는 필요하다        -> 여기서 한 번 더한다
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quantize import KeyQuant, N_PLANES, quantize_query


@dataclass(frozen=True)
class FoldedQuery:
    """K 의 per-channel scale 을 접어 넣고 양자화한 q."""

    stored: np.ndarray        # (n_steps, head_dim) int16, [-127, 127]
    scale: np.ndarray         # (n_steps, 1) — q̃ 의 양자화 스케일
    zp_correction: np.ndarray # (n_steps,) 정수. Σ_i q_st_i · z_i

    @property
    def n_steps(self) -> int:
        return self.stored.shape[0]


def fold_and_quantize_query(
    q_real: np.ndarray,
    key: KeyQuant,
    bits: int = 8,
    granularity: str = "per_token",
) -> FoldedQuery:
    """K 의 scale 을 q 에 접어 넣고 대칭 양자화한다.

    ★ "곱셈기 없음" 은 **토큰마다 반복되는 부분 내적 구간**에 대한 주장이다.
    여기의 q_real * sc 와 zero-point 보정에는 곱셈이 남아 있지만, 스텝당 한 번뿐이고
    토큰 수에 비례하지 않는다. 반복되는 쪽(masked_sum -> accumulate)이 마스킹과
    시프트 누산으로만 되어 있다는 것이 논거다.

    Parameters
    ----------
    q_real : (n_steps, head_dim) 실수 query
    key    : quantize_key() 결과
    """
    q_real = np.atleast_2d(np.asarray(q_real, dtype=np.float64))
    sc = np.asarray(key.scale, dtype=np.float64).reshape(1, -1)
    z = np.asarray(key.zero_point, dtype=np.float64).reshape(1, -1)
    if sc.shape[1] == 1:                       # per_tensor
        sc = np.broadcast_to(sc, (1, q_real.shape[1]))
        z = np.broadcast_to(z, (1, q_real.shape[1]))

    qq = quantize_query(q_real * sc, bits=bits, granularity=granularity)
    zp_corr = (qq.stored.astype(np.int64) * z.astype(np.int64)).sum(axis=-1)
    return FoldedQuery(stored=qq.stored, scale=np.asarray(qq.scale), zp_correction=zp_corr)


# ---------------------------------------------------------------------------
# 시프트 누산
# ---------------------------------------------------------------------------
def accumulate(partials: np.ndarray, m: int | None = None) -> np.ndarray:
    """S_m = Σ_{b<m} 2^(n-1-b) · P_b.  partials 는 MSB 우선 축.

    partials : (n_planes, ...)  -> 반환 (...)
    m=None 이면 전 평면 (= 정확한 점수).
    """
    p = np.asarray(partials)
    n = p.shape[0]
    m = n if m is None else int(m)

    # m 이 평면 수보다 크면 조용히 n 으로 잘려 "다 처리했다"와 구분이 안 된다
    if m > n:
        raise ValueError(f"m = {m}: only {n} planes were given")
    if m <= 0:
        return np.zeros(p.shape[1:], dtype=np.int64)
    # 시프트 누산 그대로 — 하드웨어가 하는 일을 코드도 그대로 한다.
    # tensordot 은 자리값을 곱했다. 값은 같지만 곱셈기가 없다는 주장과 어긋난다.
    out = np.zeros(p.shape[1:], dtype=np.int64)
    for b in range(n):
        out <<= 1                       # 자리 하나 올린다
        if b < m:
            out += p[b].astype(np.int64)
    return out


def cumulative_accumulate(partials: np.ndarray) -> np.ndarray:
    """모든 m 에 대한 S_m 을 한 번에. (n_planes, ...) -> (n_planes+1, ...)

    index m = 평면 m 개를 처리한 시점의 부분 점수. m=0 은 0.
    """
    p = np.asarray(partials, dtype=np.int64)
    n = p.shape[0]

    # accumulate() 와 같은 시프트 누산. 매 평면의 중간값을 남은 자리만큼 올려 둔다.
    out = np.zeros((n + 1,) + p.shape[1:], dtype=np.int64)
    running = np.zeros(p.shape[1:], dtype=np.int64)
    for b in range(n):
        running = (running << 1) + p[b]
        out[b + 1] = running << (n - b - 1)
    return out


# ---------------------------------------------------------------------------
# 최종 점수 복원
# ---------------------------------------------------------------------------
def to_real_scores(
    s_int: np.ndarray,
    fq: FoldedQuery,
    head_dim: int,
    apply_rsqrt_d: bool = True,
) -> np.ndarray:
    """정수 점수 -> 어텐션 로짓.

        s_real = scale_q · ( s_int − zp_correction )   [ / sqrt(d) ]

    s_int : (n_steps, n_tokens) 또는 (n_tokens,)
    """
    s = np.asarray(s_int, dtype=np.float64)
    corr = np.asarray(fq.zp_correction, dtype=np.float64)
    scale = np.asarray(fq.scale, dtype=np.float64).reshape(-1)
    if s.ndim == 1:
        # 1차원은 어느 스텝인지 알 수 없어 늘 0번 스텝의 보정·스케일을 썼다.
        # 스텝이 여러 개면 0번을 뺀 나머지가 전부 조용히 틀린 값이 된다.
        if corr.size != 1:
            raise ValueError(
                f"s_int is 1-D but fq holds {corr.size} steps; "
                "pass the full (n_steps, n_tokens) array so each step keeps its own scale"
            )
        out = (s - corr.reshape(-1)[0]) * scale[0]
    else:
        out = (s - corr[:, None]) * scale[:, None]
    if apply_rsqrt_d:
        out = out / np.sqrt(head_dim)
    return out


def exact_int_scores(q_stored: np.ndarray, key: KeyQuant) -> np.ndarray:
    """참값 s_int = q_st · K_st^T.  모든 설계가 이 값과 대조된다.

    (n_steps, head_dim) x (n_tokens, head_dim) -> (n_steps, n_tokens)
    """
    q = np.atleast_2d(np.asarray(q_stored, dtype=np.int64))
    return q @ key.stored.astype(np.int64).T
