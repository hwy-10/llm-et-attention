"""종단 시점 불규칙성 처리 — 배경지식 가이드 6.3-(2)절

토큰마다 종단 시점이 다르므로 파이프라인에 빈 구간이 생긴다. 세 가지 정책:

    batch       : 한 묶음 안에서 가장 늦게 끝나는 토큰에 맞춘다.
                  구현이 단순하나 절감이 작다.
    compaction  : 종단된 자리에 다음 토큰을 채워 넣는다.
                  절감이 크나 제어가 복잡하고, 압축 오버헤드가 있다.
                  ★ BRAM 워드 단위 읽기 절감을 실현하는 유일한 방법 ★
    two_phase   : 1단계에서 상위 평면만 전체 스캔, 생존 토큰만 2단계로.

각 정책은 사이클과 **BRAM 워드 읽기**를 동시에 바꾼다. 두 축을 함께 봐야
어떤 정책이 실제로 유리한지 판단할 수 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .memory import BramSpec, ReadAccount, word_reads_compacted, word_reads_scattered
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


@dataclass
class ScheduleResult:
    policy: str
    cycles: int
    reads: ReadAccount
    pipeline_efficiency: float   # 이상적 사이클 / 실제 사이클
    ideal_cycles: int            # 종단을 완벽히 활용했을 때 (= compaction 하한)

    def as_dict(self) -> dict:
        d = {
            "schedule_policy": self.policy,
            "cycles": self.cycles,
            "cycles_ideal": self.ideal_cycles,
            "pipeline_efficiency": self.pipeline_efficiency,
        }
        d.update(self.reads.as_dict())
        return d


def _ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b)) if b > 0 else 0


def baseline_cycles(n_tokens: int, spec: ScheduleSpec) -> int:
    """설계 ① 병렬 INT8 곱셈 누산 구조의 사이클."""
    return _ceil_div(n_tokens, spec.baseline_pe) * spec.baseline_cycles_per_token


def dense_cycles(n_tokens: int, n_planes: int, spec: ScheduleSpec) -> int:
    """설계 ② 비트평면 순차, 종단 없음."""
    return n_planes * _ceil_div(n_tokens, spec.lanes)


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
        n_batches = _ceil_div(n_tokens, spec.batch_size)
        cyc_per_plane_batch = _ceil_div(min(spec.batch_size, n_tokens), spec.lanes)
        for b in range(n_batches):
            sl = slice(b * spec.batch_size, min((b + 1) * spec.batch_size, n_tokens))
            planes_needed = int(rl[:, sl].any(axis=1).sum())
            cycles += planes_needed * cyc_per_plane_batch
        for t in range(n_planes):
            acc.words_bram += word_reads_scattered(rl[t], bram.word_tokens)
            acc.words_compacted += word_reads_compacted(int(rl[t].sum()), bram.word_tokens)
        acc.reads_ideal = int(rl.sum())
        ideal_cycles = sum(_ceil_div(int(rl[t].sum()), spec.lanes) for t in range(n_planes))

    elif policy == "compaction":
        # 생존 토큰을 앞으로 압축 -> 사이클과 워드 읽기가 함께 줄어든다
        for t in range(n_planes):
            n_live = int(rl[t].sum())
            cycles += _ceil_div(n_live, spec.lanes) + spec.compaction_cost_cycles
            acc.words_bram += word_reads_compacted(n_live, bram.word_tokens)
            acc.words_compacted += word_reads_compacted(n_live, bram.word_tokens)
        acc.reads_ideal = int(rl.sum())
        ideal_cycles = sum(_ceil_div(int(rl[t].sum()), spec.lanes) for t in range(n_planes))

    else:  # two_phase
        m0 = int(np.clip(spec.two_phase_split, 1, n_planes))
        # 1단계: 전체 토큰 스캔 (압축 없음)
        for t in range(m0):
            cycles += _ceil_div(n_tokens, spec.lanes)
            acc.words_bram += words_full
            acc.words_compacted += words_full
        # 압축 1회
        cycles += spec.compaction_cost_cycles
        # 2단계: 생존 토큰만 압축된 상태로
        for t in range(m0, n_planes):
            n_live = int(rl[t].sum())
            cycles += _ceil_div(n_live, spec.lanes)
            acc.words_bram += word_reads_compacted(n_live, bram.word_tokens)
            acc.words_compacted += word_reads_compacted(n_live, bram.word_tokens)
        acc.reads_ideal = int(m0 * n_tokens + rl[m0:].sum())
        ideal_cycles = m0 * _ceil_div(n_tokens, spec.lanes) + sum(
            _ceil_div(int(rl[t].sum()), spec.lanes) for t in range(m0, n_planes)
        )

    acc.bits_read = acc.words_bram * bram.word_bits
    eff = ideal_cycles / cycles if cycles > 0 else 0.0
    return ScheduleResult(
        policy=policy,
        cycles=int(cycles),
        reads=acc,
        pipeline_efficiency=float(eff),
        ideal_cycles=int(ideal_cycles),
    )


def spec_from_config(cfg, **overrides) -> ScheduleSpec:
    dp = cfg.get("hardware.datapath", {}) or {}
    sc = cfg.get("hardware.schedule", {}) or {}
    base = ScheduleSpec(
        lanes=int(dp.get("lanes", 32)),
        batch_size=int(sc.get("batch_size", 32)),
        two_phase_split=int(sc.get("two_phase_split", 3)),
        compaction_cost_cycles=int(sc.get("compaction_cost_cycles", 2)),
        baseline_pe=int(dp.get("baseline_pe", 32)),
        baseline_cycles_per_token=int(dp.get("baseline_cycles_per_token", 1)),
    )
    if overrides:
        from dataclasses import replace

        base = replace(base, **overrides)
    return base


def bram_from_config(cfg, **overrides) -> BramSpec:
    mem = cfg.get("hardware.memory", {}) or {}
    base = BramSpec(
        word_tokens=int(mem.get("word_tokens", 32)),
        word_bits=int(mem.get("word_bits", 32)),
        n_ports=int(mem.get("n_bram_ports", 2)),
        decision_latency_planes=int(mem.get("decision_latency_planes", 1)),
    )
    if overrides:
        from dataclasses import replace

        base = replace(base, **overrides)
    return base
