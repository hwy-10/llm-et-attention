"""BRAM 수행 시간 모델 — 워드를 몇 번 읽는가가 아니라 **몇 사이클 걸리는가**.

배경
----
이 저장소는 "Decode 는 메모리 병목"을 전제로 삼는다 (``src/memory.py`` 독스트링).
그런데 오랫동안 ``schedule.apply()`` 는 **연산 사이클만** 셌고, ``n_ports`` 는
``config/hardware.yaml`` → ``BramSpec`` 필드 → ``spec_from_config`` 생성자 인자,
이 셋에만 나타날 뿐 **어떤 계산에도 읽히지 않았다.** 즉

    BRAM 워드를 100번 읽어야 한다      ← 셌다
    포트가 2개니 50 사이클 걸린다       ← 세지 않았다

전제와 모델이 어긋나 있었다. 여기서 그 항을 검증한다.

★ 현재 설정이 안전한 것은 우연이다 ★
``word_tokens(32) == lanes(32)`` 이라 워드 하나가 정확히 한 연산 사이클을 채운다.
그래서 포트가 2개면 메모리가 연산의 절반이고 병목이 되지 않는다. 그런데 이 전제는
코드 어디에도 적혀 있지 않았다. ``word_tokens`` 를 줄이면 즉시 뒤집힌다
(:func:`test_narrow_words_flip_the_bottleneck` 가 그 지점을 고정한다).
"""

from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import pytest

from src.config import Config, ConfigDefaultWarning, as_bool, load_config
from src.memory import BramSpec
from src.schedule import (
    POLICIES,
    ScheduleSpec,
    apply,
    bram_from_config,
    spec_from_config,
)
from src.terminator import StepResult


def _step(read_live: np.ndarray) -> StepResult:
    """apply() 가 보는 필드만 채운 StepResult."""
    n_planes, n_tokens = read_live.shape
    return StepResult(
        s_int=np.zeros(n_tokens, dtype=np.int64),
        alive=read_live[-1].copy(),
        term_plane=read_live.sum(axis=0).astype(int),
        read_live=read_live,
        theta_trace=np.zeros(n_planes),
        live_count=read_live.sum(axis=1).astype(int),
        n_planes=n_planes,
        n_active=n_tokens,
    )


def _decaying(n_tokens=256, n_planes=8, survival=0.6, stride=1) -> np.ndarray:
    """평면이 갈수록 생존이 줄어드는 read_live. stride 로 흩뿌린다."""
    term = np.zeros(n_tokens, dtype=int)
    order = (np.arange(n_tokens) * stride) % n_tokens if stride > 1 else np.arange(n_tokens)
    for t in range(n_planes):
        n_live = n_tokens if t == 0 else max(8, int(round(n_tokens * survival**t)))
        term[order[:n_live]] += 1
    return np.arange(n_planes)[:, None] < term[None, :]


SPEC = ScheduleSpec(lanes=32, batch_size=32, two_phase_split=3, compaction_cost_cycles=2)
RL = _decaying()
SR = _step(RL)


# ---------------------------------------------------------------------------
# 산식
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("n_ports", [1, 2, 4, 8])
def test_memory_cycles_is_words_over_ports(policy, n_ports):
    """memory_cycles = ceil(words_bram / n_ports) — 산식 그대로."""
    r = apply(SR, SPEC, BramSpec(word_tokens=32, n_ports=n_ports), policy)
    assert r.memory_cycles == math.ceil(r.reads.words_bram / n_ports)


@pytest.mark.parametrize("policy", POLICIES)
def test_more_ports_never_cost_more_cycles(policy):
    """포트를 늘리면 메모리 사이클이 줄기만 해야 한다 (단조)."""
    prev = None
    for n_ports in (1, 2, 4, 8, 16):
        r = apply(SR, SPEC, BramSpec(word_tokens=32, n_ports=n_ports), policy)
        if prev is not None:
            assert r.memory_cycles <= prev
        prev = r.memory_cycles


def test_n_ports_is_no_longer_a_dead_parameter():
    """★ 회귀 방지 — n_ports 가 결과를 실제로 바꾸는지.

    예전에는 이 값을 무엇으로 바꿔도 apply() 의 출력이 한 글자도 달라지지 않았다.
    다시 그렇게 되면 여기서 막힌다.
    """
    a = apply(SR, SPEC, BramSpec(word_tokens=32, n_ports=1), "compaction")
    b = apply(SR, SPEC, BramSpec(word_tokens=32, n_ports=8), "compaction")
    assert a.memory_cycles != b.memory_cycles
    assert a.as_dict()["memory_cycles"] != b.as_dict()["memory_cycles"]


# ---------------------------------------------------------------------------
# 중첩 가정
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", POLICIES)
def test_overlap_true_is_max_false_is_sum(policy):
    """중첩 가정이 total 을 정하는 방식."""
    bram = BramSpec(word_tokens=32, n_ports=2)
    ov = apply(SR, ScheduleSpec(**{**SPEC.__dict__, "mem_overlap": True}), bram, policy)
    no = apply(SR, ScheduleSpec(**{**SPEC.__dict__, "mem_overlap": False}), bram, policy)

    assert ov.total_cycles == max(ov.cycles, ov.memory_cycles)
    assert no.total_cycles == no.cycles + no.memory_cycles
    assert no.total_cycles >= ov.total_cycles        # 보수적 가정이 더 크거나 같다


@pytest.mark.parametrize("policy", POLICIES)
def test_total_never_below_compute(policy):
    """총 시간이 연산 사이클보다 작을 수는 없다."""
    for ov in (True, False):
        r = apply(SR, ScheduleSpec(**{**SPEC.__dict__, "mem_overlap": ov}),
                  BramSpec(word_tokens=32, n_ports=4), policy)
        assert r.total_cycles >= r.cycles


def test_compute_cycles_are_unchanged_by_the_new_model():
    """★ 기존 수치가 움직이지 않았는지.

    메모리 항은 **더해서** 냈다. cycles 의 의미가 바뀌면 기존 실험 결과가
    무엇 때문에 달라졌는지 알 수 없게 된다.
    """
    for policy in POLICIES:
        base = apply(SR, SPEC, BramSpec(word_tokens=32, n_ports=2), policy)
        for n_ports in (1, 4, 8):
            for ov in (True, False):
                r = apply(SR, ScheduleSpec(**{**SPEC.__dict__, "mem_overlap": ov}),
                          BramSpec(word_tokens=32, n_ports=n_ports), policy)
                assert r.cycles == base.cycles
                assert r.reads.words_bram == base.reads.words_bram


# ---------------------------------------------------------------------------
# ★ 현재 설정이 안전한 이유와, 그것이 깨지는 지점
# ---------------------------------------------------------------------------

def test_current_config_happens_to_be_compute_bound():
    """실제 설정에서 메모리가 병목이 아니다 — 그 조건을 못 박는다.

    ★ 조건은 `word_tokens x n_ports >= lanes` 다 (2026-08-29 정정).

    한 사이클에 `lanes` 개 토큰을 먹이려면 그만큼의 토큰이 메모리에서 나와야 하고,
    한 번에 나오는 양이 `word_tokens x n_ports` 다. 이 곱이 `lanes` 에 못 미치면
    연산을 아무리 줄여도 시간이 안 준다.

    예전 판은 `word_tokens == lanes` (32 == 32) 를 전제로 삼았는데, 그건 이 조건이
    `n_ports=2` 에서 우연히 맞아떨어진 한 경우였다. `WORD_TOKENS=1` 로 바꾸면서
    `n_ports` 를 32 로 올려야 한다는 것이 드러났다 —
    1 x 2 = 2 < 32 이면 메모리가 연산의 11배가 된다.

    이 테스트가 깨지면 논문의 사이클 수치를 다시 봐야 한다.
    """
    cfg = load_config()
    spec, bram = spec_from_config(cfg), bram_from_config(cfg)
    assert bram.word_tokens * bram.n_ports >= spec.lanes, (
        f"word_tokens({bram.word_tokens}) x n_ports({bram.n_ports}) "
        f"< lanes({spec.lanes}): the premise this test rests on is broken"
    )

    for policy in POLICIES:
        r = apply(SR, spec, bram, policy)
        assert not r.memory_bound, f"{policy}: mem {r.memory_cycles} > compute {r.cycles}"
        assert r.total_cycles == r.cycles


def test_narrow_words_flip_the_bottleneck():
    """★ 워드가 레인보다 좁으면 즉시 메모리 병목이 된다.

    word_tokens=8, n_ports=1 이면 메모리 사이클이 연산의 몇 배가 된다.
    이 경우 연산을 아무리 줄여도 전체 시간이 줄지 않으므로, 조기 종단의
    사이클 절감 주장 자체가 무의미해진다.
    """
    compute_bound = apply(SR, SPEC, BramSpec(word_tokens=32, n_ports=2), "compaction")
    mem_bound = apply(SR, SPEC, BramSpec(word_tokens=8, n_ports=1), "compaction")

    assert not compute_bound.memory_bound
    assert mem_bound.memory_bound
    assert mem_bound.memory_cycles > 2 * mem_bound.cycles
    assert mem_bound.total_cycles == mem_bound.memory_cycles   # 연산은 숨는다


def test_memory_bound_flag_matches_the_numbers():
    for n_ports in (1, 2, 4):
        for wt in (4, 8, 16, 32):
            r = apply(SR, SPEC, BramSpec(word_tokens=wt, n_ports=n_ports), "compaction")
            assert r.memory_bound == (r.memory_cycles > r.cycles)


# ---------------------------------------------------------------------------
# 설정 연결
# ---------------------------------------------------------------------------

def test_config_carries_ports_and_overlap():
    """config/hardware.yaml 의 값이 실제로 모델까지 도달하는지.

    ★ 값을 맞대 보는 것만으로는 부족하다. yaml 의 n_bram_ports(2) 와
    mem_overlap(true) 이 dataclass 기본값과 **같아서**, 배선이 끊겨 기본값으로
    흘러내려도 검사식 양변이 똑같아진다 (src/config.py 의 배선 주석이 지적하는 바로
    그 함정이다). 실제로 배선 경로에 오타를 내도 예전 검사는 통과했다.
    그래서 세 가지를 본다 — 값 · 기본값으로 안 흘러내렸는지 · 다른 값도 따라오는지.
    """
    cfg = load_config()

    # ① 기본값으로 흘러내리면 경고가 나온다. 나오면 배선이 끊긴 것이다.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigDefaultWarning)
        spec, bram = spec_from_config(cfg), bram_from_config(cfg)

    # ② 값 자체. 기본값을 주지 않는다 — 키가 없으면 여기서 터져야 한다.
    assert bram.n_ports == int(cfg.get("hardware.memory.n_bram_ports"))
    assert spec.mem_overlap is as_bool(cfg.get("hardware.schedule.mem_overlap"))

    # ③ 기본값과 다른 값을 넣으면 그 값이 따라오는가
    other = Config(model=cfg.model, quant=cfg.quant, sweeps=cfg.sweeps,
                   hardware=copy.deepcopy(cfg.hardware))
    other.hardware["memory"]["n_bram_ports"] = 7
    other.hardware["schedule"]["mem_overlap"] = False
    assert bram_from_config(other).n_ports == 7, "n_ports is not read from the config"
    assert spec_from_config(other).mem_overlap is False, (
        "mem_overlap is not read from the config"
    )


def test_summary_exposes_memory_cycles():
    """as_dict() 로 새 항이 나가는지 — 실험 CSV 가 이걸 집어 간다."""
    d = apply(SR, SPEC, BramSpec(word_tokens=32, n_ports=2), "compaction").as_dict()
    for key in ("memory_cycles", "total_cycles_with_memory", "memory_bound", "mem_overlap"):
        assert key in d, key
    assert d["total_cycles_with_memory"] >= d["cycles"]


# ---------------------------------------------------------------------------
# 요약 키가 올바르게 누적되는가 — 5-2 의 마지막 항
# ---------------------------------------------------------------------------

def _summary(design, **kw):
    from experiments import build_workbench, load_config as _lc
    from src.decode_loop import run_decode
    cfg = _lc()
    wb = build_workbench(cfg)
    return run_decode(wb, design=design, top_k=8, keep_trace=False,
                      sched=spec_from_config(cfg), bram=bram_from_config(cfg), **kw).summary


def test_summary_memory_cycles_is_a_sum_of_per_step_ceilings():
    """★ 요약의 memory_cycles 는 ceil(총 워드 / 포트) 가 **아니다**.

    스텝마다 올림이 한 번씩 붙기 때문이다. 반 사이클을 다음 스텝으로 넘길 수
    없으므로 이쪽이 맞는 모델인데, 이름만 보고 총합으로 대조하면 어긋난다.
    """
    s = _summary("exact")
    n_ports = bram_from_config(load_config()).n_ports
    floor = -(-s["words_bram"] // n_ports)          # 총합 기준 하한

    assert s["total_memory_cycles"] >= floor
    # 스텝 수만큼은 더 클 수 있다 — 스텝당 최대 1 사이클씩 올림
    assert s["total_memory_cycles"] - floor <= s["n_steps"]


def test_the_baseline_row_uses_the_baseline_cycle_model():
    """★ 기준선이 자기 자신보다 8배 느리다고 나오던 자리.

    ① 은 병렬 INT8 MAC 이라 비트평면 순차와 사이클 모델이 다르다. 그런데
    apply() 가 낸 순차 사이클이 그대로 total_cycles_with_memory 에 실려
    기준선 행이 34,440 (= 연산 4,305 의 8배) 로 나왔다.
    """
    s = _summary("baseline")

    assert s["total_cycles_with_memory"] == max(s["total_cycles"], s["total_memory_cycles"])
    assert s["total_cycles_with_memory"] < 8 * s["total_cycles"]


def test_speedup_with_memory_compares_like_with_like():
    """★ 기준선을 기준선으로 나누면 두 축 모두 1.0 이어야 한다.

    분자는 연산만, 분모는 메모리 포함이면 기준선 자신이 0.25 로 나온다.
    비교 대상도 같은 축에서 재야 한다.
    """
    s = _summary("baseline")

    assert s["cycle_speedup_vs_baseline"] == pytest.approx(1.0)
    assert s["cycle_speedup_vs_baseline_with_memory"] == pytest.approx(1.0)


def test_the_memory_axis_moves_the_headline_ratio():
    """★ 축을 바꾸면 "기준선보다 8배 느리다" 가 "2배" 가 된다.

    기준선도 K 를 전부 읽어야 하므로 그쪽도 메모리 병목이다. 연산만 세면
    기준선의 읽기 시간이 통째로 빠져 격차가 부풀려진다.
    """
    base, seq = _summary("baseline"), _summary("seq")

    compute_axis = seq["total_cycles"] / base["total_cycles"]
    memory_axis = seq["total_cycles_with_memory"] / base["total_cycles_with_memory"]

    # 연산축은 평면 수라 설정과 무관하게 8.0 이다
    assert compute_axis == pytest.approx(8.0, rel=0.02)

    # ★ 메모리축 값은 설정에 의존한다 (2026-08-29 정정)
    #     word_tokens=32, n_ports=2   -> 2.00   기준선 메모리 17,220
    #     word_tokens=1,  n_ports=32  -> 1.05   기준선 메모리 32,760
    #   포트가 늘면 기준선의 읽기 시간도 함께 줄어 격차가 더 좁혀진다.
    #   고정값 대신 "격차가 줄어든다" 는 성질을 못 박는다.
    assert memory_axis < compute_axis / 2, (
        f"memory_axis={memory_axis:.3f} — 축을 바꿔도 격차가 안 줄었다"
    )
    assert 1.0 <= memory_axis <= 2.5, f"memory_axis={memory_axis:.3f} 가 범위 밖이다"


def test_the_baseline_memory_quota_is_dense_not_the_test_designs_reads():
    """★ 기준선의 메모리 몫은 dense 워드다 — 제안 설계의 절감을 빌려오면 안 된다.

    ① 은 K 를 전부 읽으므로 종단 절감이 없다. 그런데 기준선의 메모리 몫을
    `words_dense` 가 아니라 그 스텝이 실제로 읽은 `words_bram` 으로 재면,
    기준선이 조기 종단의 절감을 공짜로 가져가 **제안 설계가 더 나빠 보인다**
    (exact 의 메모리축 속도비 0.5223 -> 0.3872).

    baseline / seq 는 정책이 none 이라 두 값이 같아 이 착오가 드러나지 않는다.
    종단이 일어나는 설계에서만 갈리므로 여기서 못 박는다.
    """
    base, exact = _summary("baseline"), _summary("exact")

    # 같은 워크로드의 기준선은 설계와 무관하게 같은 값이어야 한다
    assert exact["total_baseline_cycles_with_memory"] == base["total_cycles_with_memory"]

    # 종단이 실제로 읽기를 줄였는데도 기준선 몫은 안 줄어야 한다
    assert exact["total_memory_cycles"] < base["total_memory_cycles"], "no termination savings"
    assert exact["total_baseline_cycles_with_memory"] > exact["total_memory_cycles"]
