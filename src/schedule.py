"""종단 시점 불규칙성 처리 — 배경지식 가이드 6.3-(2)절

토큰마다 종단 시점이 다르므로 파이프라인에 빈 구간이 생긴다. 세 가지 정책:

    batch       : 한 묶음 안에서 가장 늦게 끝나는 토큰에 맞춘다.
                  구현이 단순하나 절감이 작다.
    compaction  : 종단된 자리에 다음 토큰을 채워 넣는다.
                  절감이 크나 제어가 복잡하고, 압축 오버헤드가 있다.
                  ★ BRAM 워드 단위 읽기 절감을 실현하는 유일한 방법 ★
    two_phase   : 1단계에서 상위 평면만 전체 스캔, 생존 토큰만 2단계로.

각 정책은 사이클과 BRAM 워드 읽기를 동시에 바꾼다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConfigDefaultWarning, apply_overrides, as_bool, read_fields
from .memory import (
    BramSpec,
    ReadAccount,
    _ceil_div,
    word_reads_compacted,
    word_reads_scattered,
)
from .terminator import StepResult

POLICIES = ("none", "batch", "compaction", "two_phase")


@dataclass(frozen=True)
class ScheduleSpec:
    lanes: int = 32                  # 평면 하나에서 한 사이클에 처리하는 토큰 수
    batch_size: int = 32
    two_phase_split: int = 3
    compaction_cost_cycles: int = 2  # 평면당 압축 오버헤드
    baseline_pe: int = 32
    baseline_cycles_per_token: int = 1
    # True: total = max(연산, 메모리) / False: total = 연산 + 메모리
    # RTL 실측으로 확정할 가정 (config/hardware.yaml: source=estimate)
    mem_overlap: bool = True


@dataclass
class ScheduleResult:
    policy: str
    cycles: int                  # 연산 사이클 (의미 불변)
    reads: ReadAccount
    pipeline_efficiency: float   # 이상적 사이클 / 실제 사이클
    ideal_cycles: int            # 종단을 완벽히 활용했을 때
    memory_cycles: int = 0       # ceil(words_bram / n_ports)
    total_cycles: int = 0        # mem_overlap 가정에 따라 max 또는 합
    mem_overlap: bool = True

    @property
    def memory_bound(self) -> bool:
        # True 면 연산을 더 줄여도 시간이 안 준다
        return self.memory_cycles > self.cycles

    def as_dict(self) -> dict:
        d = {
            "schedule_policy": self.policy,
            "cycles": self.cycles,
            "cycles_ideal": self.ideal_cycles,
            "pipeline_efficiency": self.pipeline_efficiency,
            "memory_cycles": self.memory_cycles,
            "total_cycles_with_memory": self.total_cycles,
            "memory_bound": self.memory_bound,
            "mem_overlap": self.mem_overlap,
        }
        d.update(self.reads.as_dict())
        return d


def baseline_cycles(n_tokens: int, spec: ScheduleSpec) -> int:
    """설계 ① 병렬 INT8 곱셈 누산 구조의 사이클."""
    per_pass = _ceil_div(n_tokens, spec.baseline_pe, what="baseline_pe")
    return per_pass * spec.baseline_cycles_per_token


def dense_cycles(n_tokens: int, n_planes: int, spec: ScheduleSpec) -> int:
    """설계 ② 비트평면 순차, 종단 없음."""
    return n_planes * _ceil_div(n_tokens, spec.lanes, what="lanes")


def apply(
    result: StepResult,
    spec: ScheduleSpec,
    bram: BramSpec,
    policy: str = "compaction",
) -> ScheduleResult:
    """종단 결과에 스케줄 정책을 적용해 사이클과 읽기를 산출한다."""
    if policy not in POLICIES:
        raise KeyError(f"unknown schedule policy {policy!r}; choose from {POLICIES}")

    rl = result.read_live                      # (n_planes, n_tokens)
    n_planes, n_tokens = rl.shape
    words_full = bram.n_words(n_tokens)

    acc = ReadAccount(
        reads_dense=n_planes * n_tokens,
        words_dense=n_planes * words_full,
    )
    ideal_cycles = 0
    cycles = 0

    if policy == "none":
        # 종단을 아예 쓰지 않는다 (설계 ②)
        cycles = dense_cycles(n_tokens, n_planes, spec)
        acc.reads_ideal = n_planes * n_tokens
        acc.words_bram = acc.words_dense
        acc.words_compacted = acc.words_dense
        ideal_cycles = cycles

    elif policy == "batch":
        # 묶음 단위로 가장 늦게 끝나는 토큰에 맞춘다.
        # 묶음이 살아 있으면 그 안의 죽은 토큰까지 함께 끌고 간다 —
        # 사이클과 워드를 같은 기준으로 세야 한다.
        n_batches = _ceil_div(n_tokens, spec.batch_size, what="batch_size")
        dragged = np.zeros_like(rl)
        for b in range(n_batches):
            lo = b * spec.batch_size
            hi = min(lo + spec.batch_size, n_tokens)
            alive = rl[:, lo:hi].any(axis=1)            # 이 묶음이 살아 있는 평면
            # 마지막 묶음은 폭이 좁다. 꽉 찬 묶음과 같은 값을 청구하지 않는다.
            cycles += int(alive.sum()) * _ceil_div(hi - lo, spec.lanes, what="lanes")
            dragged[alive, lo:hi] = True
        for t in range(n_planes):
            acc.words_bram += word_reads_scattered(dragged[t], bram.word_tokens)
            acc.words_compacted += word_reads_compacted(int(rl[t].sum()), bram.word_tokens)
        acc.reads_ideal = int(rl.sum())
        ideal_cycles = sum(
            _ceil_div(int(rl[t].sum()), spec.lanes, what="lanes") for t in range(n_planes)
        )

    elif policy == "compaction":
        # 생존 토큰을 앞으로 압축 -> 사이클과 워드 읽기가 함께 줄어든다
        for t in range(n_planes):
            n_live = int(rl[t].sum())
            cycles += _ceil_div(n_live, spec.lanes, what="lanes") + spec.compaction_cost_cycles
            acc.words_bram += word_reads_compacted(n_live, bram.word_tokens)
            acc.words_compacted += word_reads_compacted(n_live, bram.word_tokens)
        acc.reads_ideal = int(rl.sum())
        ideal_cycles = sum(
            _ceil_div(int(rl[t].sum()), spec.lanes, what="lanes") for t in range(n_planes)
        )

    else:  # two_phase
        m0 = int(np.clip(spec.two_phase_split, 1, n_planes))
        # 1단계: 전체 토큰 스캔 (압축 없음)
        for t in range(m0):
            cycles += _ceil_div(n_tokens, spec.lanes, what="lanes")
            acc.words_bram += words_full
            acc.words_compacted += words_full
        # 압축 1회 — 이 시점의 생존 토큰을 앞으로 모은다
        cycles += spec.compaction_cost_cycles
        # 2단계: 압축된 배열을 훑는다.
        # 압축이 한 번뿐이므로 그 뒤에 죽는 토큰은 배열 안에 구멍으로 남는다.
        # 매 평면 압축한 것처럼 세면 compaction 의 절감을 공짜로 가져가게 된다.
        if m0 < n_planes:
            order = np.flatnonzero(rl[m0])          # 압축 배열에 들어가는 토큰
            slot = np.full(n_tokens, -1, dtype=np.int64)
            slot[order] = np.arange(order.size)

            for t in range(m0, n_planes):
                idx = slot[np.flatnonzero(rl[t])]
                if idx.size and int(idx.min()) < 0:
                    raise ValueError(
                        "two_phase: read_live is not monotone, so a token outside the "
                        f"compacted set is still read at plane {t}"
                    )
                packed = np.zeros(order.size, dtype=bool)
                packed[idx] = True

                # 죽은 덩어리는 건너뛴다. 구멍은 그대로 끌고 간다.
                cycles += word_reads_scattered(packed, spec.lanes)
                acc.words_bram += word_reads_scattered(packed, bram.word_tokens)
                acc.words_compacted += word_reads_compacted(
                    int(rl[t].sum()), bram.word_tokens
                )
        acc.reads_ideal = int(m0 * n_tokens + rl[m0:].sum())
        ideal_cycles = m0 * _ceil_div(n_tokens, spec.lanes, what="lanes") + sum(
            _ceil_div(int(rl[t].sum()), spec.lanes, what="lanes") for t in range(m0, n_planes)
        )

    acc.bits_read = acc.words_bram * bram.word_bits
    eff = ideal_cycles / cycles if cycles > 0 else 0.0

    # 메모리 사이클 — 포트 수만큼 한 사이클에 읽는다
    mem_cycles = _ceil_div(acc.words_bram, bram.n_ports, what="n_ports")
    total = max(int(cycles), mem_cycles) if spec.mem_overlap else int(cycles) + mem_cycles

    return ScheduleResult(
        policy=policy,
        cycles=int(cycles),
        reads=acc,
        pipeline_efficiency=float(eff),
        ideal_cycles=int(ideal_cycles),
        memory_cycles=int(mem_cycles),
        total_cycles=int(total),
        mem_overlap=bool(spec.mem_overlap),
    )


# config/hardware.yaml -> dataclass 배선표. (필드, yaml 점표기, 변환)
SPEC_WIRING = (
    ("lanes",                     "hardware.datapath.lanes",                     int),
    ("batch_size",                "hardware.schedule.batch_size",                int),
    ("two_phase_split",           "hardware.schedule.two_phase_split",           int),
    ("compaction_cost_cycles",    "hardware.schedule.compaction_cost_cycles",    int),
    ("baseline_pe",               "hardware.datapath.baseline_pe",               int),
    ("baseline_cycles_per_token", "hardware.datapath.baseline_cycles_per_token", int),
    ("mem_overlap",               "hardware.schedule.mem_overlap",               as_bool),
)

BRAM_WIRING = (
    ("word_tokens",             "hardware.memory.word_tokens",             int),
    ("word_bits",               "hardware.memory.word_bits",               int),
    ("n_ports",                 "hardware.memory.n_bram_ports",            int),
    ("decision_latency_planes", "hardware.memory.decision_latency_planes", int),
)


def spec_from_config(cfg, **overrides) -> ScheduleSpec:
    """config/hardware.yaml -> ScheduleSpec."""

    # 설정 파싱 (누락 항목: 기본값 대체 + ConfigDefaultWarning)
    fields = read_fields(cfg, ScheduleSpec, SPEC_WIRING)

    return apply_overrides(ScheduleSpec(**fields), overrides)


def bram_from_config(cfg, **overrides) -> BramSpec:
    """config/hardware.yaml -> BramSpec."""

    # 설정 파싱 (누락 항목: 기본값 대체 + ConfigDefaultWarning)
    fields = read_fields(cfg, BramSpec, BRAM_WIRING)

    return apply_overrides(BramSpec(**fields), overrides)
