"""조기 종단이 하드웨어 스케줄에 어떻게 반영되는지 눈으로 확인한다.

★ 이 모듈은 스케줄을 **다시 구현하지 않는다.** ★

    from src import schedule
    schedule.apply(result, spec, bram, policy)

실제 :mod:`src.schedule` 을 그대로 부른다. 교보재가 본체를 흉내 내면 둘이
갈라져도 아무도 모르고, 그러면 화면이 설명하는 것은 본체가 아니라 흉내가 된다.
여기가 하는 일은 입력(``read_live``)을 만들어 넣고 결과를 JSON 으로 펴는 것뿐이다.

무엇을 보여 주려는가
--------------------
``src/memory.py`` 의 경고가 이 데모의 주제다 — **BRAM 은 토큰을 하나씩 읽지 않는다.**
워드 하나에 ``word_tokens`` 개 토큰이 묶여 있어서, 그 안에 살아있는 토큰이
하나라도 있으면 워드 전체를 읽어야 한다. 따라서

    토큰 50% 종단  ≠  사이클 50% 감소  ≠  BRAM 읽기 50% 감소

이다. 이걸 보이려고 ``clustering`` 손잡이를 뒀다. **생존 토큰의 개수는 그대로 두고
배치만 바꾼다** — ``reads_ideal`` 은 한 톨도 변하지 않는데 ``words_bram`` 은
크게 움직인다. 그 격차가 work-compaction 이 필요한 이유다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src import schedule as _schedule
from src.memory import BramSpec, word_reads_compacted, word_reads_scattered
from src.terminator import StepResult

__all__ = [
    "LIMITS", "MAX_TRACE_SLOTS", "build_trace", "hardware_defaults",
    "make_read_live", "run",
]

# 화면과 응답 크기를 감당할 수 있는 범위. 계산 자체의 제약은 아니다.
#
# 상한은 이 프로젝트의 실제 스펙을 덮도록 잡았다 — config/hardware.yaml 은
# lanes=32, word_tokens=32, batch_size=32 이고, KV 캐시 길이 T 는
# architecture.md 기준 512 까지 본다. 그 값들이 슬라이더 한가운데 오도록
# 상한을 그 위로 둔다.
LIMITS: dict[str, tuple[int, int]] = {
    "n_tokens": (8, 512),      # 디코드 스텝의 활성 토큰 수 = 문맥 길이 T
    "n_planes": (2, 16),       # INT8 이면 8. 더 넓은 양자화도 볼 수 있게 16 까지
    "lanes": (1, 256),         # 실제 32
    "batch_size": (1, 512),    # 실제 32
    "word_tokens": (1, 256),   # 실제 32
    "word_bits": (8, 512),     # 실제 32
    "n_ports": (1, 16),        # 실제 2 — BRAM 포트 수. 메모리 사이클을 정한다
    "two_phase_split": (1, 16),
    "compaction_cost_cycles": (0, 16),
    "baseline_pe": (1, 256),
}

# 타임라인 DOM 이 감당할 수 있는 슬롯 수. 넘으면 트레이스를 생략하고 알린다.
# 실제 스펙(512토큰 · 8평면 · 32레인 -> 약 4,100 슬롯)은 넉넉히 들어간다.
MAX_TRACE_SLOTS = 12000

_PHI = 0.6180339887498949  # 황금비 켤레 — 저불일치 수열을 만들어 최대로 흩뜨린다


def _clamp_int(value: Any, key: str) -> int:
    lo, hi = LIMITS[key]
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} is not an integer: {value!r}") from None
    return max(lo, min(hi, v))


def _clamp_unit(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return float(min(1.0, max(0.0, v)))


# ---------------------------------------------------------------------------
# 입력 만들기
# ---------------------------------------------------------------------------

def make_read_live(
    n_tokens: int,
    n_planes: int,
    survival: float = 0.62,
    clustering: float = 0.0,
    keep_frac: float = 0.08,
) -> np.ndarray:
    """평면별로 "아직 읽어야 하는 토큰"을 나타내는 (n_planes, n_tokens) bool 배열.

    두 손잡이가 서로 **독립**이라는 점이 핵심이다.

    ``survival``
        평면이 하나 넘어갈 때마다 살아남는 비율. 평면별 생존 **수**를 정한다.
    ``clustering``
        그 생존 토큰들을 어디에 놓을지만 정한다. 0이면 최대로 흩뿌리고,
        1이면 앞쪽에 뭉친다. **생존 수는 전혀 바뀌지 않는다.**

    그래서 clustering 만 움직이면 ``reads_ideal`` 은 고정인 채
    ``words_bram`` 만 변한다 — 그 격차가 이 데모가 보여 주려는 것이다.
    """
    survival = float(min(1.0, max(0.0, survival)))
    clustering = float(min(1.0, max(0.0, clustering)))

    # 평면별 생존 수 (평면 0 은 언제나 전부 읽는다 — 아직 판정할 근거가 없다)
    keep = max(1, int(round(n_tokens * keep_frac)))
    live_count = [n_tokens]
    for t in range(1, n_planes):
        live_count.append(max(keep, int(round(n_tokens * survival**t))))

    # 각 토큰이 몇 장까지 읽히는가 — 위 생존 수열이 개수를 이미 결정한다
    term_plane = np.zeros(n_tokens, dtype=int)
    idx = np.arange(n_tokens)

    # 배치 손잡이: 뭉침(단조) 과 흩뿌림(황금비 저불일치) 을 섞는다
    mono = idx / max(1, n_tokens - 1)
    spread = (idx * _PHI) % 1.0
    key = clustering * mono + (1.0 - clustering) * spread
    order = np.argsort(key, kind="stable")  # key 가 작은 토큰이 오래 산다

    for t in range(n_planes):
        term_plane[order[: live_count[t]]] += 1

    planes = np.arange(n_planes)[:, None]
    return planes < term_plane[None, :]


def _as_step_result(read_live: np.ndarray) -> StepResult:
    """``schedule.apply`` 가 쓰는 필드만 채운 최소 StepResult.

    apply() 는 ``read_live`` 하나만 본다. 나머지는 모양을 맞춘 자리 채우기다.
    """
    n_planes, n_tokens = read_live.shape
    term_plane = read_live.sum(axis=0).astype(int)
    return StepResult(
        s_int=np.zeros(n_tokens, dtype=np.int64),
        alive=read_live[-1].copy(),
        term_plane=term_plane,
        read_live=read_live,
        theta_trace=np.zeros(n_planes, dtype=float),
        live_count=read_live.sum(axis=1).astype(int),
        n_planes=n_planes,
        n_active=n_tokens,
    )


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def run(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """설정 하나에 대해 네 정책을 모두 돌려 JSON 으로 펼 수 있는 dict 를 만든다."""
    p = dict(params or {})

    n_tokens = _clamp_int(p.get("n_tokens", 96), "n_tokens")
    n_planes = _clamp_int(p.get("n_planes", 8), "n_planes")
    survival = _clamp_unit(p.get("survival", 0.62), 0.62)
    clustering = _clamp_unit(p.get("clustering", 0.0), 0.0)

    spec = _schedule.ScheduleSpec(
        lanes=_clamp_int(p.get("lanes", 32), "lanes"),
        batch_size=_clamp_int(p.get("batch_size", 32), "batch_size"),
        two_phase_split=_clamp_int(p.get("two_phase_split", 3), "two_phase_split"),
        compaction_cost_cycles=_clamp_int(
            p.get("compaction_cost_cycles", 2), "compaction_cost_cycles"
        ),
        baseline_pe=_clamp_int(p.get("baseline_pe", 32), "baseline_pe"),
        baseline_cycles_per_token=1,
        # 연산과 BRAM 읽기가 겹치는가 — 검증되지 않은 가정이므로 화면에서 바꿔 본다
        mem_overlap=bool(p.get("mem_overlap", True)),
    )
    bram = BramSpec(
        word_tokens=_clamp_int(p.get("word_tokens", 32), "word_tokens"),
        word_bits=_clamp_int(p.get("word_bits", 32), "word_bits"),
        n_ports=_clamp_int(p.get("n_ports", 2), "n_ports"),
        decision_latency_planes=1,
    )

    read_live = make_read_live(n_tokens, n_planes, survival, clustering)
    result = _as_step_result(read_live)

    # ★ 여기가 전부다 — 실제 src.schedule 을 그대로 부른다
    policies = {}
    traces = {}
    for name in _schedule.POLICIES:
        res = _schedule.apply(result, spec, bram, name)
        policies[name] = res.as_dict()

        cyc = build_trace(read_live, spec, name)
        # ★ 그림이 본체를 설명하지 못하면 그리지 않는다
        if len(cyc) != res.cycles:
            raise AssertionError(
                f"{name}: trace has {len(cyc)} cycles != apply() {res.cycles}"
            )
        if len(cyc) * spec.lanes > MAX_TRACE_SLOTS:
            traces[name] = {"cycles": None, "n_cycles": len(cyc),
                            "occupancy": _occupancy(cyc, spec.lanes),
                            "omitted": True}
        else:
            traces[name] = {"cycles": cyc, "n_cycles": len(cyc),
                            "occupancy": _occupancy(cyc, spec.lanes),
                            "omitted": False}

    # 평면별 내역 — 화면의 주인공 그림이 쓴다
    per_plane = []
    for t in range(n_planes):
        row = read_live[t]
        n_live = int(row.sum())
        per_plane.append(
            {
                "plane": t,
                "live": n_live,
                "words_scattered": word_reads_scattered(row, bram.word_tokens),
                "words_compacted": word_reads_compacted(n_live, bram.word_tokens),
                "cycles_lanes": int(np.ceil(n_live / spec.lanes)) if spec.lanes else 0,
            }
        )

    return {
        "read_live": read_live.astype(int).tolist(),
        "term_plane": result.term_plane.tolist(),
        "per_plane": per_plane,
        "policies": policies,
        "traces": traces,
        "verdict": _verdict(policies, traces, {
            "reads_ideal": int(read_live.sum()),
            "reads_dense": n_planes * n_tokens,
        }, n_planes, spec.compaction_cost_cycles),
        "hardware_defaults": hardware_defaults(),
        "reference": {
            "baseline_cycles": _schedule.baseline_cycles(n_tokens, spec),
            "dense_cycles": _schedule.dense_cycles(n_tokens, n_planes, spec),
            "words_full": bram.n_words(n_tokens),
            "words_dense": n_planes * bram.n_words(n_tokens),
            "reads_dense": n_planes * n_tokens,
            "reads_ideal": int(read_live.sum()),
            "n_words": bram.n_words(n_tokens),
        },
        "spec": {
            "n_tokens": n_tokens,
            "n_planes": n_planes,
            "survival": survival,
            "clustering": clustering,
            "lanes": spec.lanes,
            "batch_size": spec.batch_size,
            "two_phase_split": min(spec.two_phase_split, n_planes),
            "compaction_cost_cycles": spec.compaction_cost_cycles,
            "baseline_pe": spec.baseline_pe,
            "word_tokens": bram.word_tokens,
            "word_bits": bram.word_bits,
            "n_ports": bram.n_ports,
            "mem_overlap": spec.mem_overlap,
        },
        "policy_order": list(_schedule.POLICIES),
    }


# ---------------------------------------------------------------------------
# 실행 트레이스 — 사이클 하나하나를 그릴 수 있게 편다
# ---------------------------------------------------------------------------
#
# apply() 는 사이클 **수**만 돌려준다. 화면에 스케줄을 그리려면 그 사이클들이
# 각각 무엇을 하는지 알아야 한다. 아래 함수는 apply() 안의 식을 그대로 따라가며
# 슬롯을 채우고, **마지막에 총 사이클 수가 apply() 와 같은지 확인한다.**
# 어긋나면 그림이 본체를 설명하지 못하므로 그 자리에서 예외를 던진다.


def _emit(cycles, read_live, lanes, plane, kind, tokens, batch=None):
    """사이클 한 줄을 만든다.

    ``slots[i]`` 는 레인 i 에 실린 토큰 번호이거나 None(빈 슬롯)이다.
    ``waste[i]`` 는 **이미 종단됐는데 끌려온** 토큰을 뜻한다 — 정책이 종단을
    회수하지 못하고 있다는 표시이고, 화면에서 이게 눈에 보여야 한다.
    """
    slots = list(tokens[:lanes])
    waste = [
        bool(plane is not None and tok is not None and not read_live[plane][tok])
        for tok in slots
    ]
    pad = lanes - len(slots)
    slots += [None] * pad
    waste += [False] * pad
    cycles.append(
        {
            "index": len(cycles),
            "plane": plane,
            "kind": kind,
            "batch": batch,
            "slots": slots,
            "waste": waste,
        }
    )


def build_trace(read_live: np.ndarray, spec, policy: str) -> list[dict[str, Any]]:
    """정책 하나의 사이클별 레인 점유를 만든다. apply() 의 식을 그대로 따른다."""
    n_planes, n_tokens = read_live.shape
    lanes = max(1, spec.lanes)
    cycles: list[dict[str, Any]] = []

    def chunks(seq):
        for c in range(0, max(len(seq), 0), lanes):
            yield seq[c : c + lanes]

    if policy == "none":
        # 종단을 무시한다 — 죽은 토큰도 그대로 레인을 차지한다
        for t in range(n_planes):
            for part in chunks(list(range(n_tokens))):
                _emit(cycles, read_live, lanes, t, "compute", part)

    elif policy == "batch":
        n_batches = int(np.ceil(n_tokens / spec.batch_size))
        for b in range(n_batches):
            lo = b * spec.batch_size
            hi = min((b + 1) * spec.batch_size, n_tokens)
            toks = list(range(lo, hi))
            # apply() 와 동일: 살아있는 토큰이 하나라도 있는 평면만 센다
            planes = [t for t in range(n_planes) if read_live[t, lo:hi].any()]
            for t in planes:
                # 묶음의 실제 폭으로 나눈다 — 마지막 묶음은 그만큼만 돈다
                for part in chunks(toks):
                    _emit(cycles, read_live, lanes, t, "compute", part, batch=b)

    elif policy == "compaction":
        for t in range(n_planes):
            live = np.flatnonzero(read_live[t]).tolist()
            for _ in range(spec.compaction_cost_cycles):
                _emit(cycles, read_live, lanes, None, "compaction", [], batch=t)
            for part in chunks(live):
                _emit(cycles, read_live, lanes, t, "compute", part)

    else:  # two_phase
        m0 = int(np.clip(spec.two_phase_split, 1, n_planes))
        for t in range(m0):                       # 1단계 — 전체 스캔
            for part in chunks(list(range(n_tokens))):
                _emit(cycles, read_live, lanes, t, "compute", part)
        for _ in range(spec.compaction_cost_cycles):   # 압축 1회
            _emit(cycles, read_live, lanes, None, "compaction", [], batch=m0)
        if m0 < n_planes:
            # 압축은 한 번뿐이다. 뒤에 죽는 토큰은 배열 안에 구멍으로 남는다.
            order = np.flatnonzero(read_live[m0]).tolist()
            for t in range(m0, n_planes):
                for part in chunks(order):
                    # 통째로 죽은 덩어리만 건너뛴다
                    if any(read_live[t][tok] for tok in part):
                        _emit(cycles, read_live, lanes, t, "compute", part)

    return cycles


def _occupancy(cycles: list[dict[str, Any]], lanes: int) -> dict[str, Any]:
    """레인 슬롯이 얼마나 쓰였는지. "파이프라인에 빈 구간이 생긴다"의 수치화."""
    total = len(cycles) * lanes
    useful = waste = 0
    for cyc in cycles:
        for tok, w in zip(cyc["slots"], cyc["waste"]):
            if tok is None:
                continue
            if w:
                waste += 1
            else:
                useful += 1
    return {
        "slots_total": total,
        "slots_useful": useful,        # 살아있는 토큰을 실제로 처리한 슬롯
        "slots_waste": waste,          # 종단된 토큰을 끌고 간 슬롯
        "slots_empty": total - useful - waste,
        "utilization": (useful / total) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# 어느 정책이 유리한가 — 두 축이 갈릴 수 있으므로 한 줄로 답하지 않는다
# ---------------------------------------------------------------------------

def hardware_defaults() -> dict[str, Any]:
    """``config/hardware.yaml`` 의 실제 스펙을 읽어 온다.

    데모가 손으로 베낀 숫자를 들고 있으면 인터페이스 파일이 바뀔 때 조용히
    낡는다. 그래서 프리셋 "실제 스펙" 은 여기서 직접 읽는다.
    설정 파일을 못 읽으면 dataclass 기본값으로 되돌아간다.
    """
    fallback = {
        "lanes": 32, "word_tokens": 32, "batch_size": 32,
        "two_phase_split": 3, "compaction_cost_cycles": 2,
        "n_ports": 2, "mem_overlap": True,
        "n_planes": 8, "source": "dataclass 기본값 (config 를 읽지 못함)",
    }
    try:
        from src.config import load_config

        cfg = load_config()
        spec = _schedule.spec_from_config(cfg)
        bram = _schedule.bram_from_config(cfg)
        return {
            "lanes": spec.lanes,
            "word_tokens": bram.word_tokens,
            "batch_size": spec.batch_size,
            "two_phase_split": spec.two_phase_split,
            "compaction_cost_cycles": spec.compaction_cost_cycles,
            "n_ports": bram.n_ports,
            "mem_overlap": spec.mem_overlap,
            "n_planes": int(cfg.get("quant.planes.n_planes", 8) or 8),
            "source": "config/hardware.yaml",
        }
    except Exception:  # 설정이 없거나 깨져도 데모는 떠야 한다
        return fallback


_SITUATIONS = {
    "no_gain": (
        "회수할 종단이 거의 없다",
        "어느 정책을 써도 BRAM 워드가 의미 있게 줄지 않는다. 압축은 비용만 남기므로 "
        "제어가 가장 단순한 쪽이 낫다.",
    ),
    "clustered": (
        "종단이 뭉쳐서 일어난다",
        "죽은 토큰이 같은 워드에 모여 있어 재배치 없이도 워드를 통째로 건너뛸 수 있다. "
        "batch 가 compaction 만큼 얻으므로 압축 비용을 낼 이유가 없다.",
    ),
    "scattered": (
        "종단이 흩어져서 일어난다",
        "워드마다 살아있는 토큰이 남아, 재배치 없이는 워드를 건너뛸 수 없다. "
        "compaction 이 워드 절감을 얻는 유일한 방법이다.",
    ),
    "scattered_costly": (
        "흩어져 죽지만, 매 평면 압축은 비싸다",
        "two_phase 가 compaction 의 워드 절감을 거의 그대로 유지하면서 사이클은 더 적다. "
        "압축을 한 번만 내기 때문이다.",
    ),
}


def _verdict(policies, traces, reference, n_planes, compaction_cost) -> dict[str, Any]:
    """지금 설정에서 어느 정책이 어느 축에서 이기는지 판정한다.

    **한 정책을 '정답'으로 내놓지 않는다.** 사이클과 BRAM 워드는 서로 다른 축이고
    실제로 자주 갈린다 — 그때는 갈렸다고 말하는 것이 맞는 답이다.

    판정은 임계값 감각이 아니라 **실제로 계산된 결과**로 한다. "종단률이 몇 % 면
    무슨 상황" 같은 규칙은 lanes·word_tokens 가 바뀌면 바로 틀리기 때문이다.
    """
    order = list(policies)
    best_cycles = min(order, key=lambda n: (policies[n]["cycles"], order.index(n)))
    best_words = min(order, key=lambda n: (policies[n]["words_bram"], order.index(n)))
    best_util = max(
        order, key=lambda n: (traces[n]["occupancy"]["utilization"], -order.index(n))
    )

    w_batch = policies["batch"]["read_saving_bram"]
    w_comp = policies["compaction"]["read_saving_bram"]
    w_two = policies["two_phase"]["read_saving_bram"]
    c_comp = policies["compaction"]["cycles"]
    c_two = policies["two_phase"]["cycles"]

    if w_comp < 0.05:
        # 압축을 해도 워드가 안 줄어든다 -> 회수할 것이 없다
        key = "no_gain"
        rec = "none" if policies["none"]["cycles"] <= policies["batch"]["cycles"] else "batch"
    elif w_comp - w_batch <= 0.03:
        # 재배치 없이도 같은 절감을 얻는다 -> 제어가 단순한 쪽
        key, rec = "clustered", "batch"
    elif c_two < c_comp and w_two >= 0.9 * w_comp:
        # 압축 1회만으로 절감 대부분을 유지하면서 사이클이 더 적다
        key, rec = "scattered_costly", "two_phase"
    else:
        key, rec = "scattered", "compaction"

    label, why = _SITUATIONS[key]
    # ★ 메모리가 병목이면 연산 절감은 시간으로 돌아오지 않는다
    mem_bound = {n: policies[n]["memory_bound"] for n in order}
    best_total = min(order, key=lambda n: (policies[n]["total_cycles_with_memory"],
                                           order.index(n)))
    if all(mem_bound.values()):
        key, rec = "memory_bound", best_total
        label = "메모리가 병목이다"
        why = ("BRAM 읽기 사이클이 연산 사이클을 넘는다. 이 구간에서는 연산을 더 줄여도 "
               "전체 시간이 줄지 않으므로, 사이클이 아니라 워드를 줄이는 정책을 골라야 한다. "
               "포트를 늘리거나 word_tokens 를 키우면 벗어난다.")

    return {
        "best_cycles": best_cycles,
        "best_words": best_words,
        "best_total": best_total,
        "any_memory_bound": any(mem_bound.values()),
        "all_memory_bound": all(mem_bound.values()),
        "best_utilization": best_util,
        "axes_agree": best_cycles == best_words,
        "situation": key,
        "situation_label": label,
        "why": why,
        "recommended": rec,
        "termination_frac": (
            1.0 - reference["reads_ideal"] / reference["reads_dense"]
            if reference["reads_dense"] else 0.0
        ),
        "saving_batch": w_batch,
        "saving_compaction": w_comp,
        "saving_two_phase": w_two,
        "compaction_overhead_frac": (compaction_cost * n_planes / c_comp) if c_comp else 0.0,
    }
