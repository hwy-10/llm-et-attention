"""함수 하나를 입력 → 핵심 동작 → 출력 상자로 펼친다.

★ 상자는 손으로 베끼지 않는다 ★

매개변수 목록은 :func:`inspect.signature`, 필드 목록은
:func:`dataclasses.fields`, 정책 목록은 ``schedule.POLICIES`` 에서 **실제 코드를
읽어** 만든다. 설명 문장만 사람이 쓴다.

이렇게 하는 이유는 그림이 조용히 낡는 것을 막기 위해서다. 손으로 베낀 상자는
누가 필드를 추가해도 그대로 남아 있어서, 보는 사람은 그림이 최신이라고 믿는다.
여기서는 코드에 필드가 늘면 상자도 늘고, 설명이 없는 항목은 화면에
**\"설명 없음\"** 으로 뜬다 — 어긋남이 바로 보인다.

지금은 :func:`src.schedule.apply` 하나만 펼친다. 나머지 함수는 목록에만 올리고
``implemented: False`` 로 표시한다.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

from src import memory as _memory
from src import schedule as _schedule
from src import terminator as _terminator

__all__ = ["MODULES", "describe", "module_index"]


# ---------------------------------------------------------------------------
# 사람이 쓰는 부분 — 설명 문장
# ---------------------------------------------------------------------------
# 키는 **실제 이름**과 맞춰야 한다. 코드에 있는데 여기 없으면 화면에
# "설명 없음" 으로 뜨고, 여기 있는데 코드에 없으면 아예 안 그려진다.

_TYPE_DESC: dict[str, str] = {
    "StepResult": "종단 판정 결과. 이 함수가 실제로 보는 것은 `read_live` 하나뿐이다.",
    "ScheduleSpec": "데이터패스 구성. 레인 수·묶음 크기·압축 비용 등 **연산 쪽** 설정.",
    "BramSpec": "BRAM 구성. 워드 하나에 토큰 몇 개가 묶이는지 등 **메모리 쪽** 설정.",
    "ScheduleResult": "사이클과 읽기 회계. **점수는 들어 있지 않다** — 스케줄은 답을 바꾸지 않는다.",
    "ReadAccount": "읽기 회계. 이론값과 실현값을 **반드시 함께** 낸다.",
}

_FIELD_DESC: dict[tuple[str, str], str] = {
    # --- StepResult (입력) — apply() 가 쓰는 것만 표시한다
    ("StepResult", "read_live"): (
        "★ **이 함수가 쓰는 유일한 필드.** (n_planes, n_tokens) bool. "
        "`read_live[t][i]` 는 평면 t 에서 토큰 i 를 아직 읽어야 하는가."
    ),
    ("StepResult", "s_int"): "정수 점수. apply() 는 **읽지 않는다.**",
    ("StepResult", "alive"): "끝까지 살아남은 토큰. apply() 는 읽지 않는다.",
    ("StepResult", "term_plane"): "토큰별 종단 평면 수. apply() 는 읽지 않는다.",
    ("StepResult", "theta_trace"): "평면별 θ. apply() 는 읽지 않는다.",
    ("StepResult", "live_count"): "평면별 논리적 활성 수. apply() 는 읽지 않는다.",
    ("StepResult", "n_planes"): "평면 수. `read_live.shape` 에서 다시 구하므로 apply() 는 읽지 않는다.",
    ("StepResult", "n_active"): "이 스텝의 활성 토큰 수. apply() 는 읽지 않는다.",
    ("StepResult", "extra"): "부가 정보. apply() 는 읽지 않는다.",
    # --- ScheduleSpec
    ("ScheduleSpec", "lanes"): "한 사이클에 처리하는 토큰 수. `ceil(토큰 / lanes)` 로 평면당 사이클이 정해진다.",
    ("ScheduleSpec", "batch_size"): "`batch` 정책의 묶음 크기. 다른 정책은 쓰지 않는다.",
    ("ScheduleSpec", "two_phase_split"): "`two_phase` 의 1단계 평면 수 m0. 다른 정책은 쓰지 않는다.",
    ("ScheduleSpec", "compaction_cost_cycles"): (
        "압축 오버헤드. `compaction` 은 **평면마다**, `two_phase` 는 **한 번만** 낸다."
    ),
    ("ScheduleSpec", "baseline_pe"): "기준 설계(①)의 PE 수. apply() 안에서는 쓰지 않는다.",
    ("ScheduleSpec", "baseline_cycles_per_token"): "기준 설계의 토큰당 사이클. apply() 안에서는 쓰지 않는다.",
    ("ScheduleSpec", "mem_overlap"): (
        "연산과 메모리가 겹치는가. True 면 `max`, False 면 합. "
        "**RTL 실측으로 확정할 가정이다.**"
    ),
    # --- BramSpec
    ("BramSpec", "word_tokens"): (
        "★ 워드 하나에 묶이는 토큰 수. **절감의 알갱이 크기.** "
        "워드 안에 살아있는 토큰이 하나라도 있으면 워드 전체를 읽는다."
    ),
    ("BramSpec", "word_bits"): "워드 하나의 비트 수. `bits_read` 를 만들 때만 쓴다.",
    ("BramSpec", "n_ports"): "한 사이클에 읽을 수 있는 워드 수. `memory_cycles` 를 정한다.",
    ("BramSpec", "decision_latency_planes"): (
        "판정 지연. **apply() 는 쓰지 않는다** — `read_live` 를 만들 때 이미 반영되어 들어온다."
    ),
    # --- ScheduleResult (출력)
    ("ScheduleResult", "policy"): "적용한 정책 이름. 입력을 그대로 되돌려 준다.",
    ("ScheduleResult", "cycles"): "★ **연산 사이클.** 정책마다 세는 방식이 다르다.",
    ("ScheduleResult", "reads"): "읽기 회계 묶음 (`ReadAccount`). 아래 필드 참조.",
    ("ScheduleResult", "pipeline_efficiency"): (
        "`ideal_cycles / cycles`. 1.0 에 가까울수록 종단 이득을 잘 회수한 것이다."
    ),
    ("ScheduleResult", "ideal_cycles"): (
        "오버헤드 없이 생존 토큰만 완벽히 꽉 채웠을 때의 하한. "
        "`Σ ceil(live[t] / lanes)`."
    ),
    ("ScheduleResult", "memory_cycles"): "`ceil(words_bram / n_ports)`. BRAM 읽기가 먹는 사이클.",
    ("ScheduleResult", "total_cycles"): "연산과 메모리를 합친 값. `mem_overlap` 가정에 따라 max 또는 합.",
    ("ScheduleResult", "mem_overlap"): "어떤 가정으로 `total_cycles` 를 냈는지 기록해 둔 것.",
    # --- ReadAccount
    ("ReadAccount", "reads_ideal"): "살아있는 (토큰, 평면) 쌍의 수. **이론적 절감**의 분자.",
    ("ReadAccount", "reads_dense"): "종단이 없을 때의 (토큰, 평면) 쌍 = `n_planes × n_tokens`.",
    ("ReadAccount", "words_bram"): (
        "★ **실제로 읽어야 하는 BRAM 워드 수.** 정책마다 scattered / compacted 로 갈린다. "
        "여기가 이 함수의 핵심 출력이다."
    ),
    ("ReadAccount", "words_dense"): "종단이 없을 때의 워드 수 = `n_planes × ceil(n_tokens / word_tokens)`.",
    ("ReadAccount", "words_compacted"): "압축했을 때의 워드 수 — 달성 가능한 하한.",
    ("ReadAccount", "bits_read"): "`words_bram × word_bits`. 마지막에 한 번 계산한다.",
}

#: apply() 안의 네 갈래. 코드 인용은 실제 소스에서 잘라 온다 (아래 _branch_source).
_BRANCH_DESC: dict[str, dict[str, str]] = {
    "none": {
        "title": "종단을 아예 쓰지 않는다 (설계 ②)",
        "what": (
            "`dense_cycles()` 를 그대로 쓰고, 워드도 `words_dense` 를 그대로 넣는다. "
            "`read_live` 를 **한 번도 보지 않는다** — 그래서 종단이 얼마나 일어났든 결과가 같다."
        ),
        "why": "비교 기준점이다. 다른 세 정책이 여기서 얼마나 줄였는지를 재는 자다.",
    },
    "batch": {
        "title": "묶음 안에서 가장 늦게 끝나는 토큰에 맞춘다",
        "what": (
            "묶음별로 `rl[:, sl].any(axis=1).sum()` — 그 묶음에 살아있는 토큰이 "
            "하나라도 있는 평면의 수를 센다. 워드는 `word_reads_scattered` 로, "
            "즉 **재배치 없이 있는 그대로** 읽는다."
        ),
        "why": (
            "제어가 가장 단순하다. 하지만 종단이 흩어져 있으면 워드가 거의 안 줄어든다 — "
            "실제 파이프라인 데이터에서는 `none` 과 사이클·워드가 **완전히 같았다.**"
        ),
    },
    "compaction": {
        "title": "생존 토큰을 앞으로 당긴다",
        "what": (
            "평면마다 `n_live = rl[t].sum()` 을 세고 `ceil(n_live / lanes)` 사이클을 쓴다. "
            "워드는 `word_reads_compacted(n_live, ...)` — **재배치했다고 가정**한 값이다. "
            "대신 평면마다 `compaction_cost_cycles` 를 더한다."
        ),
        "why": (
            "★ **BRAM 워드 절감을 실현하는 유일한 방법.** 배치와 무관하게 같은 값을 낸다. "
            "다만 종단이 적으면 압축 비용만 남아 손해가 된다."
        ),
    },
    "two_phase": {
        "title": "앞은 전체 스캔, 뒤는 압축된 상태",
        "what": (
            "`m0 = clip(two_phase_split, 1, n_planes)` 장까지는 `rl` 을 보지 않고 "
            "`words_full` 을 그대로 더한다. 그 뒤 **압축을 한 번만** 하고, "
            "남은 평면은 생존 토큰만 읽는다."
        ),
        "why": (
            "압축 비용을 한 번만 내는 절충안이다. `compaction` 의 워드 절감을 "
            "대부분 유지하면서 사이클이 더 적은 경우가 많다."
        ),
    },
}

_CORE_STEPS: list[dict[str, str]] = [
    {
        "id": "guard",
        "title": "① 정책 이름 검사",
        "code": 'if policy not in POLICIES:\n    raise KeyError(...)',
        "detail": (
            "오타를 조용히 넘기지 않는다. 알 수 없는 정책이면 `KeyError` 를 던진다. "
            "기본값이 `\"compaction\"` 이라 인자를 빼먹으면 압축 정책이 적용된다."
        ),
    },
    {
        "id": "setup",
        "title": "② 공통 회계 준비",
        "code": (
            "rl = result.read_live            # (n_planes, n_tokens)\n"
            "words_full = bram.n_words(n_tokens)\n"
            "acc = ReadAccount(\n"
            "    reads_dense=n_planes * n_tokens,\n"
            "    words_dense=n_planes * words_full,\n"
            ")"
        ),
        "detail": (
            "`reads_dense` 와 `words_dense` 는 **정책과 무관한 분모**다. "
            "모든 절감률이 이 두 값을 기준으로 계산되므로 여기가 틀리면 전부 틀린다."
        ),
    },
    {
        "id": "branch",
        "title": "③ 정책별 분기 — 네 갈래",
        "code": 'if policy == "none": ...\nelif policy == "batch": ...\nelif policy == "compaction": ...\nelse:  # two_phase',
        "detail": (
            "★ 네 가지가 갈리는 지점은 **생존 토큰을 재배치하느냐** 하나다. "
            "아래 정책 상자를 눌러 각각을 보라."
        ),
    },
    {
        "id": "finish",
        "title": "④ 마무리 — 비트 수와 효율",
        "code": (
            "acc.bits_read = acc.words_bram * bram.word_bits\n"
            "eff = ideal_cycles / cycles if cycles > 0 else 0.0\n"
            "mem_cycles = _ceil_div(acc.words_bram, bram.n_ports)\n"
            "total = max(cycles, mem_cycles) if spec.mem_overlap else cycles + mem_cycles"
        ),
        "detail": (
            "네 갈래가 끝난 뒤 공통으로 도는 부분이다. "
            "`memory_cycles` 는 **오랫동안 빠져 있던 항**으로, `n_ports` 가 "
            "설정에만 있고 어떤 계산에도 읽히지 않던 죽은 값이었다."
        ),
    },
]


# ---------------------------------------------------------------------------
# 코드에서 읽어 오는 부분
# ---------------------------------------------------------------------------

def _fields_of(cls: type, owner: str) -> list[dict[str, Any]]:
    """dataclass 의 필드를 상자 항목으로 편다. 설명이 없으면 그렇다고 표시한다."""
    out: list[dict[str, Any]] = []
    for f in dataclasses.fields(cls):
        desc = _FIELD_DESC.get((owner, f.name))
        default: Any = None
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default = "(factory)"
        out.append(
            {
                "name": f.name,
                "type": _type_name(f.type),
                "default": None if default is None else str(default),
                "desc": desc,
                "documented": desc is not None,
            }
        )
    return out


def _type_name(t: Any) -> str:
    if isinstance(t, str):
        return t
    return getattr(t, "__name__", str(t))


def _branch_source(policy: str) -> str:
    """apply() 원본에서 해당 분기만 잘라 온다 (손으로 베끼지 않는다)."""
    src = inspect.getsource(_schedule.apply).splitlines()
    marks = {
        "none": ('if policy == "none":', 'elif policy == "batch":'),
        "batch": ('elif policy == "batch":', 'elif policy == "compaction":'),
        "compaction": ('elif policy == "compaction":', "else:  # two_phase"),
        "two_phase": ("else:  # two_phase", "acc.bits_read"),
    }
    start_mark, end_mark = marks[policy]
    lo = hi = None
    for i, line in enumerate(src):
        s = line.strip()
        if lo is None and s.startswith(start_mark):
            lo = i
        elif lo is not None and s.startswith(end_mark):
            hi = i
            break
    if lo is None:
        return "(원본에서 이 분기를 찾지 못했습니다 — 코드가 바뀌었는지 확인하세요)"
    block = src[lo : hi if hi is not None else len(src)]
    while block and not block[-1].strip():
        block.pop()
    indent = min((len(l) - len(l.lstrip()) for l in block if l.strip()), default=0)
    return "\n".join(l[indent:] for l in block)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

MODULES: dict[str, Any] = {"schedule": _schedule}


def module_index(module: str = "schedule") -> dict[str, Any]:
    """모듈의 최상위 이름을 목록으로 낸다. 지금은 apply() 만 펼쳐져 있다."""
    if module not in MODULES:
        raise KeyError(f"unknown module {module!r}; choose from {sorted(MODULES)}")
    mod = MODULES[module]

    implemented = {"apply"}
    blurb = {
        "apply": "종단 결과에 스케줄 정책을 적용해 사이클과 읽기를 산출한다",
        "baseline_cycles": "설계 ① 병렬 INT8 MAC 구조의 사이클",
        "dense_cycles": "설계 ② 비트평면 순차, 종단 없음",
        "spec_from_config": "config/hardware.yaml → ScheduleSpec",
        "bram_from_config": "config/hardware.yaml → BramSpec",
        "ScheduleSpec": "데이터패스 구성 (dataclass)",
        "ScheduleResult": "사이클·읽기 회계 결과 (dataclass)",
    }

    items: list[dict[str, Any]] = []
    for name, obj in vars(mod).items():
        if name.startswith("_"):
            continue
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        items.append(
            {
                "name": name,
                "kind": "class" if inspect.isclass(obj) else "function",
                "signature": _signature_of(obj),
                "blurb": blurb.get(name, ""),
                "implemented": name in implemented,
                "lineno": inspect.getsourcelines(obj)[1],
            }
        )
    items.sort(key=lambda d: d["lineno"])
    return {
        "module": f"src/{module}.py",
        "n_lines": len(inspect.getsource(mod).splitlines()),
        "items": items,
    }


def _signature_of(obj: Any) -> str:
    try:
        return f"{obj.__name__}{inspect.signature(obj)}"
    except (TypeError, ValueError):
        return obj.__name__


def describe(module: str = "schedule", func: str = "apply") -> dict[str, Any]:
    """함수 하나를 입력 → 핵심 동작 → 출력 상자로 편다."""
    if module != "schedule" or func != "apply":
        raise KeyError(
            f"{module}.{func} is not expanded yet (only schedule.apply for now)"
        )

    sig = inspect.signature(_schedule.apply)
    type_map = {
        "result": _terminator.StepResult,
        "spec": _schedule.ScheduleSpec,
        "bram": _memory.BramSpec,
    }

    inputs: list[dict[str, Any]] = []
    for name, p in sig.parameters.items():
        cls = type_map.get(name)
        tname = cls.__name__ if cls is not None else _type_name(p.annotation)
        entry: dict[str, Any] = {
            "name": name,
            "type": tname,
            "default": None if p.default is inspect.Parameter.empty else repr(p.default),
            "desc": _TYPE_DESC.get(tname, ""),
            "fields": _fields_of(cls, tname) if cls is not None else [],
        }
        if name == "policy":
            entry["desc"] = "네 가지 중 하나. 이 값이 아래 분기를 고른다."
            entry["choices"] = list(_schedule.POLICIES)
        inputs.append(entry)

    branches = [
        {
            "policy": p,
            "title": _BRANCH_DESC[p]["title"],
            "what": _BRANCH_DESC[p]["what"],
            "why": _BRANCH_DESC[p]["why"],
            "code": _branch_source(p),
        }
        for p in _schedule.POLICIES
    ]

    outputs = [
        {
            "name": "ScheduleResult",
            "type": "dataclass",
            "desc": _TYPE_DESC["ScheduleResult"],
            "fields": _fields_of(_schedule.ScheduleResult, "ScheduleResult"),
        },
        {
            "name": "ReadAccount",
            "type": "dataclass (ScheduleResult.reads)",
            "desc": _TYPE_DESC["ReadAccount"],
            "fields": _fields_of(_memory.ReadAccount, "ReadAccount"),
        },
    ]

    undocumented = [
        f"{owner}.{f['name']}"
        for owner, group in (
            ("ScheduleSpec", inputs[1]["fields"]),
            ("BramSpec", inputs[2]["fields"]),
            ("StepResult", inputs[0]["fields"]),
            ("ScheduleResult", outputs[0]["fields"]),
            ("ReadAccount", outputs[1]["fields"]),
        )
        for f in group
        if not f["documented"]
    ]

    return {
        "module": "src/schedule.py",
        "func": "apply",
        "signature": _signature_of(_schedule.apply),
        "doc": inspect.getdoc(_schedule.apply) or "",
        "lineno": inspect.getsourcelines(_schedule.apply)[1],
        "n_lines": len(inspect.getsource(_schedule.apply).splitlines()),
        "inputs": inputs,
        "core": _CORE_STEPS,
        "branches": branches,
        "outputs": outputs,
        # ★ 코드에 있는데 설명이 없는 항목. 비어 있어야 정상이다.
        "undocumented": undocumented,
    }
