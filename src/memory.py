"""비트평면 BRAM 모델 + 메모리 읽기 회계 — 배경지식 가이드 5.7 / 6.2절

가이드 5.7절: 조기 종단으로 줄어드는 것은 (1) 연산 사이클 (2) 메모리 읽기이며,
Decode 는 메모리 병목이므로 **(2)가 더 중요하다.**

★★ 그런데 여기에 함정이 있다 ★★

BRAM 은 토큰을 하나씩 읽지 않는다. 한 워드에 word_tokens 개 토큰의
같은 평면 비트가 묶여 있다.

    워드 안에 살아있는 토큰이 하나라도 있으면 -> 워드 전체를 읽어야 한다

따라서 종단된 토큰이 흩어져 있으면 읽기는 거의 줄지 않는다.
이 파일은 두 수치를 **반드시 함께** 낸다:

    reads_ideal : 살아있는 (토큰, 평면) 쌍의 수          <- 이론적 절감
    reads_bram  : 실제로 읽어야 하는 BRAM 워드 수        <- 실현되는 절감

두 값의 격차가 schedule.py 의 work-compaction 이 필요한 진짜 이유다.
논문에 reads_ideal 만 쓰면 발표에서 바로 지적당한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import read_fields


def _ceil_div(a: int, b: int, *, what: str = "나눌 폭") -> int:
    # a = 세는 값(토큰·워드), b = 한 번에 처리하는 폭
    # 0 을 돌려주면 잘못된 설정이 "읽기 0" 으로 조용히 묻힌다
    if b <= 0:
        raise ValueError(f"{what} = {b}: must be >= 1 (tried to divide {a})")
    if a < 0:
        raise ValueError(f"count = {a}: must not be negative ({what} = {b})")
    return -(-int(a) // int(b))   # 정수 연산 — float 반올림을 안 탄다


@dataclass(frozen=True)
class BramSpec:
    """비트평면 BRAM 구성."""

    # ★ 기본값은 일부러 config/hardware.yaml 과 다르게 둔다.
    #   둘이 같으면 배선이 끊겨도 숫자가 안 변해 알 수 없다
    #   (read_fields 의 ConfigDefaultWarning 이 그 대비책이다).
    #   실제 값은 config 가 정한다. -> docs/architecture.md §4.1.1
    word_tokens: int = 32       # 한 워드에 담기는 토큰 수 (평면 1개 기준)
    # 워드 하나의 물리 폭. bits_read 를 만들 때만 쓴다.
    # 토큰 하나가 평면당 1비트이므로 word_bits >= word_tokens 여야 뜻이 선다.
    word_bits: int = 32
    n_ports: int = 2
    decision_latency_planes: int = 1
    # ★ 판정 지연은 상수가 아니다 — ARCHITECTURE.md 6.2.1
    #   "fixed" 는 위 상수를 그대로 쓴다 (기존 수치 재현용)
    #   "auto"  는 스텝마다 토큰 수로부터 계산한다 (이쪽이 물리적으로 맞다)
    decision_latency_mode: str = "fixed"
    pipeline_cycles: int = 8    # 가산트리 6단 + 마스킹 1 + 비교기 1

    def n_words(self, n_tokens: int) -> int:
        return _ceil_div(n_tokens, self.word_tokens, what="word_tokens")

    def latency_at(self, n_tokens: int, lanes: int) -> int:
        """이 스텝에서 실제로 적용될 판정 지연(평면 수)."""
        if self.decision_latency_mode != "auto":
            return int(self.decision_latency_planes)
        return latency_planes(n_tokens, lanes, self.pipeline_cycles)


def latency_planes(n_tokens: int, lanes: int, pipeline_cycles: int = 8) -> int:
    """판정 지연을 사이클에서 **평면 수**로 환산한다.

        지연(평면) = ceil( pipeline_cycles / ceil(n_tokens / lanes) )

    ★ 상수가 아니라 문맥 길이의 함수다. lanes=32, pipeline=8 기준

        n_tokens =  32 -> 8 평면   (지연이 전 평면. 종단 이득이 0)
        n_tokens = 128 -> 2 평면
        n_tokens >= 256 -> 1 평면

    고정값 1 은 n_tokens >= 256 에서만 맞다. 짧은 문맥에서 이 보정을 빼면
    읽기 절감이 최대 2배 과대평가된다 (T=128 에서 20.9% -> 10.7%).
    """
    if lanes <= 0 or n_tokens <= 0 or pipeline_cycles <= 0:
        return 0
    return _ceil_div(pipeline_cycles, _ceil_div(n_tokens, lanes, what="lanes"),
                     what="cycles_per_plane")


def word_reads_scattered(live_row: np.ndarray, word_tokens: int) -> int:
    """살아있는 토큰이 흩어져 있을 때 읽어야 하는 워드 수.

    워드 안에 살아있는 토큰이 하나라도 있으면 그 워드를 읽는다.
    """
    live = np.asarray(live_row, dtype=bool)

    # 평면 하나의 행을 받는다. 2차원을 통째로 넘기면 평면 경계를 넘어
    # 한 워드로 묶여 조용히 다른 값이 나온다.
    if live.ndim != 1:
        raise ValueError(
            f"live_row must be 1-D (one plane), got shape {live.shape}; "
            "pass read_live[plane], not the whole array"
        )
    if word_tokens <= 0:
        raise ValueError(f"word_tokens = {word_tokens}: must be >= 1")

    n = live.size
    pad = (-n) % word_tokens
    if pad:
        live = np.concatenate([live, np.zeros(pad, dtype=bool)])
    return int(live.reshape(-1, word_tokens).any(axis=1).sum())


def word_reads_compacted(n_live: int, word_tokens: int) -> int:
    """살아있는 토큰을 앞으로 압축했을 때의 워드 수."""
    return _ceil_div(int(n_live), word_tokens, what="word_tokens")


@dataclass
class ReadAccount:
    """한 디코드 스텝의 읽기 회계."""

    reads_ideal: int = 0          # 살아있는 (토큰, 평면) 쌍
    reads_dense: int = 0          # 종단이 없을 때의 (토큰, 평면) 쌍
    words_bram: int = 0           # 실제 BRAM 워드 읽기 (흩어진 배치)
    words_dense: int = 0          # 종단이 없을 때의 워드 읽기
    words_compacted: int = 0      # 압축했을 때의 워드 읽기 (달성 가능 하한)
    bits_read: int = 0

    @property
    def ideal_saving(self) -> float:
        """이론적 읽기 절감률."""
        return 1.0 - self.reads_ideal / self.reads_dense if self.reads_dense else 0.0

    @property
    def bram_saving(self) -> float:
        """★ 실현되는 읽기 절감률 (워드 단위) ★"""
        return 1.0 - self.words_bram / self.words_dense if self.words_dense else 0.0

    @property
    def compacted_saving(self) -> float:
        """압축을 적용했을 때 달성 가능한 절감률."""
        return 1.0 - self.words_compacted / self.words_dense if self.words_dense else 0.0

    @property
    def realization_ratio(self) -> float:
        """실현률 = 실현 절감 / 이론 절감. 1.0 이면 완전히 실현된 것."""
        return self.bram_saving / self.ideal_saving if self.ideal_saving > 0 else 1.0

    def as_dict(self) -> dict:
        return {
            "reads_ideal": self.reads_ideal,
            "reads_dense": self.reads_dense,
            "words_bram": self.words_bram,
            "words_dense": self.words_dense,
            "words_compacted": self.words_compacted,
            "bits_read": self.bits_read,
            "read_saving_ideal": self.ideal_saving,
            "read_saving_bram": self.bram_saving,
            "read_saving_compacted": self.compacted_saving,
            "read_realization_ratio": self.realization_ratio,
        }

    def __iadd__(self, other: "ReadAccount") -> "ReadAccount":
        self.reads_ideal += other.reads_ideal
        self.reads_dense += other.reads_dense
        self.words_bram += other.words_bram
        self.words_dense += other.words_dense
        self.words_compacted += other.words_compacted
        self.bits_read += other.bits_read
        return self


def account_step(read_live: np.ndarray, spec: BramSpec) -> ReadAccount:
    """한 스텝의 읽기 회계를 계산한다.

    read_live : (n_planes, n_tokens) bool — terminator.run_step() 출력
    """
    rl = np.asarray(read_live, dtype=bool)
    n_planes, n_tokens = rl.shape
    words_full = spec.n_words(n_tokens)

    acc = ReadAccount(
        reads_ideal=int(rl.sum()),
        reads_dense=int(n_planes * n_tokens),
        words_dense=int(n_planes * words_full),
    )
    for t in range(n_planes):
        acc.words_bram += word_reads_scattered(rl[t], spec.word_tokens)
        acc.words_compacted += word_reads_compacted(int(rl[t].sum()), spec.word_tokens)
    acc.bits_read = acc.words_bram * spec.word_bits
    return acc


# ---------------------------------------------------------------------------
# 용량 점검 (가이드 3.5절)
# ---------------------------------------------------------------------------
def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int,
                   seq_len: int, dtype_bytes: int = 2) -> int:
    """외부 KV 캐시 전체 크기 (K+V, FP16). 가이드 3.5절의 16 MB 계산."""
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes * seq_len


def bitplane_bram_bytes(n_tokens: int, head_dim: int, n_planes: int = 8) -> int:
    """비트평면으로 저장한 K 하나(1층 1헤드)의 온칩 용량."""
    return n_tokens * head_dim * n_planes // 8


@dataclass(frozen=True)
class ModelShape:
    """용량 계산에 쓰는 모델 치수."""

    head_dim: int = 64
    n_layers: int = 16
    n_kv_heads: int = 8
    kv_dtype_bytes: int = 2


# config/model.yaml -> dataclass 배선표. (필드, yaml 점표기, 변환)
MODEL_WIRING = (
    ("head_dim",       "model.model.head_dim",       int),
    ("n_layers",       "model.model.n_layers",       int),
    ("n_kv_heads",     "model.model.n_kv_heads",     int),
    ("kv_dtype_bytes", "model.model.kv_dtype_bytes", int),
)


def capacity_report(cfg, seq_len: int | None = None) -> dict:
    """온칩에 무엇이 들어가고 무엇이 안 들어가는지 정리."""

    # 설정 파싱 (누락 항목: 기본값 대체 + ConfigDefaultWarning)
    shape = ModelShape(**read_fields(cfg, ModelShape, MODEL_WIRING))

    seq_len = seq_len or cfg.seq_len
    per_head = bitplane_bram_bytes(seq_len, shape.head_dim, cfg.n_planes)

    return {
        "seq_len": seq_len,
        "kv_cache_external_bytes": kv_cache_bytes(
            shape.n_layers, shape.n_kv_heads, shape.head_dim,
            seq_len, shape.kv_dtype_bytes,
        ),
        "k_bitplane_one_head_bytes": per_head,
        "k_bitplane_all_heads_bytes": per_head * shape.n_layers * shape.n_kv_heads,
    }
