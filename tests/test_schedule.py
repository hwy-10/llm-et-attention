"""스케줄 정책 검증 — 배경지식 가이드 6.3-(2)

정책은 사이클과 읽기를 바꾸지만 **점수는 절대 바꾸지 않는다**.

뒷부분(설정 배선)은 ``spec_from_config`` / ``bram_from_config`` 를 다룬다.
두 함수 모두 이 모듈의 함수이므로 여기서 같이 검증한다.
"""

import shutil
import warnings

from dataclasses import replace

import numpy as np
import pytest

from src.accumulator import exact_int_scores, fold_and_quantize_query
from src.bounds import step_bounds
from src.config import CONFIG_DIR, CONFIG_FILES, ConfigDefaultWarning, load_config
from src.designs import run_design
from src.masked_sum import partial_dots
from src.memory import BramSpec, word_reads_compacted
from src.quantize import quantize_key, to_bitplanes
from src.terminator import StepResult
from src.schedule import (
    BRAM_WIRING,
    POLICIES,
    SPEC_WIRING,
    ScheduleSpec,
    apply,
    baseline_cycles,
    bram_from_config,
    dense_cycles,
    spec_from_config,
)


def _case(seed=0, d=64, T=256):
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 1.0, size=(1, d))
    k = rng.normal(0, 1.0, size=(T, d))
    k[0] = 4.0 * (q[0] / np.linalg.norm(q[0])) * np.linalg.norm(k[0])
    key = quantize_key(k)
    fq = fold_and_quantize_query(q, key)
    p = partial_dots(fq.stored, to_bitplanes(key.stored, 8))[:, 0, :]
    return p, step_bounds(fq.stored[0]), exact_int_scores(fq.stored, key)[0]


SCHED = ScheduleSpec(lanes=32, batch_size=32, two_phase_split=3, compaction_cost_cycles=2)
BRAM = BramSpec(word_tokens=32, decision_latency_planes=1)


def test_dense_and_baseline_cycles():
    # 토큰 256, lanes 32 -> 평면당 8사이클, 8평면 -> 64
    assert dense_cycles(256, 8, SCHED) == 64
    # 기준 설계: PE 32개, 토큰당 1사이클 -> 8사이클
    assert baseline_cycles(256, SCHED) == 8


def test_a_zero_width_raises_instead_of_returning_zero():
    """폭이 0 이하일 때 사이클 0 으로 묻히지 않는지 검증."""

    # lanes = 0 — 어느 항목이 잘못됐는지 메시지에 있어야 한다
    with pytest.raises(ValueError, match="lanes"):
        dense_cycles(256, 8, ScheduleSpec(lanes=0))

    # baseline_pe = 0 — 같은 함수를 쓰지만 다른 이름이 나와야 한다
    with pytest.raises(ValueError, match="baseline_pe"):
        baseline_cycles(256, ScheduleSpec(baseline_pe=0))

    # 음수도 마찬가지
    with pytest.raises(ValueError, match="lanes"):
        dense_cycles(256, 8, ScheduleSpec(lanes=-32))


def test_a_negative_count_raises():
    """세는 값이 음수일 때 조용히 넘어가지 않는지 검증."""

    with pytest.raises(ValueError, match="negative"):
        dense_cycles(-1, 8, SCHED)

    # 0 개는 정상이다 — 종단으로 전멸한 평면이 여기 해당한다
    assert dense_cycles(0, 8, SCHED) == 0


def test_ceil_is_exact_for_huge_counts():
    """정수 나눗셈이므로 float 정밀도 한계를 넘어도 어긋나지 않는다."""

    huge = 2**53 + 1                      # float 로는 2**53 과 구분되지 않는 값

    assert dense_cycles(huge, 1, ScheduleSpec(lanes=1)) == huge


def test_bitserial_is_structurally_slower_than_baseline():
    """비트평면 순차가 기준 설계의 정확히 n_planes 배인지 검증."""

    n_planes = load_config().n_planes
    spec = ScheduleSpec(lanes=32, baseline_pe=32, baseline_cycles_per_token=1)

    for t in (32, 100, 256, 257, 512):
        dense, base = dense_cycles(t, n_planes, spec), baseline_cycles(t, spec)

        # 부등식이 아니라 등식 — 2배로 줄어드는 버그도 잡는다
        assert dense == n_planes * base, f"T={t}: dense={dense}, baseline={base}"

        # 제안 설계가 느리다는 사실 자체 (n_planes > 1 이므로 따라 나온다)
        assert dense > base


def test_the_ratio_follows_parallelism_not_the_plane_count():
    """비율을 정하는 것이 평면 수가 아니라 병렬도임을 검증."""

    # "8배" 는 lanes == baseline_pe 일 때만이다. 평면 수는 내내 8 이다.
    # 두 축을 각각 흔든다 — 한쪽만 흔들면 다른 쪽을 상수로 박아도 통과한다.
    cases = [
        # lanes  baseline_pe  기대 비율 = 8 * baseline_pe / lanes
        (16, 32, 16),
        (32, 32, 8),
        (64, 32, 4),
        (32, 64, 16),
        (32, 16, 4),
    ]

    for lanes, baseline_pe, ratio in cases:
        spec = ScheduleSpec(lanes=lanes, baseline_pe=baseline_pe,
                            baseline_cycles_per_token=1)
        dense, base = dense_cycles(256, 8, spec), baseline_cycles(256, spec)

        # dense 가 lanes 를, baseline 이 baseline_pe 를 읽는지 여기서 갈린다
        assert dense == ratio * base, (
            f"lanes={lanes} baseline_pe={baseline_pe}: dense={dense}, baseline={base}"
        )


def test_baseline_cycles_per_token_divides_the_ratio():
    """토큰당 사이클이 비율에 들어가는지 검증."""

    # 기준 PE 가 토큰당 2사이클이면 8배가 아니라 4배다
    for cpt, ratio in ((1, 8), (2, 4), (4, 2)):
        spec = ScheduleSpec(lanes=32, baseline_pe=32, baseline_cycles_per_token=cpt)
        dense, base = dense_cycles(256, 8, spec), baseline_cycles(256, spec)

        assert dense == ratio * base, f"cpt={cpt}: dense={dense}, baseline={base}"


def test_the_real_config_still_supports_the_8x_claim():
    """문서의 8배가 성립하는 전제가 실제 설정에서 유지되는지 감시."""

    cfg = load_config()
    spec = spec_from_config(cfg)

    # 이 둘이 갈리면 문서의 8배가 그 순간 틀린 말이 된다
    assert spec.lanes == spec.baseline_pe, f"{spec.lanes} != {spec.baseline_pe}"
    assert spec.baseline_cycles_per_token == 1, spec.baseline_cycles_per_token

    assert dense_cycles(256, cfg.n_planes, spec) == cfg.n_planes * baseline_cycles(256, spec)


def test_policy_does_not_change_scores():
    """정책이 답을 건드리지 않는지 검증 — 점수·생존·읽기 마스크 전부."""

    for design in ("exact", "approx"):
        p, b, _ = _case(0)
        ref = None

        for pol in POLICIES:
            r = run_design(design, p, b, top_k=8, schedule_policy=pol,
                           sched=SCHED, bram=BRAM)
            # scores_raw 만 보면 손실 허용 스케줄러가 들어와도 못 잡는다
            cur = (r.scores_raw.tobytes(), r.scores.tobytes(), r.alive.tobytes(),
                   r.term_plane.tobytes(), r.step_result.read_live.tobytes())

            if ref is None:
                ref = cur
            else:
                assert cur == ref, f"{design}/{pol}"


def test_decision_latency_changes_reads_but_not_scores():
    """정책보다 넓은 불변식 — 하드웨어 설정이 답을 바꾸지 않는다."""

    p, b, _ = _case(0)
    out = {}
    for latency in (0, 3):
        r = run_design("exact", p, b, top_k=8, schedule_policy="compaction",
                       sched=SCHED, bram=BramSpec(word_tokens=32,
                                                  decision_latency_planes=latency))
        out[latency] = (r.scores_raw.tobytes(), int(r.step_result.read_live.sum()))

    # 점수는 같고
    assert out[0][0] == out[3][0]

    # 읽을 양만 는다
    assert out[3][1] > out[0][1]


# ---------------------------------------------------------------------------
# 정책 판별 — 손으로 셀 수 있는 작은 케이스
#
# 4평면 x 16토큰, lanes=4, word_tokens=4, batch_size=4, two_phase_split=2.
# 평면별 생존 수를 [16, 16, 8, 4] 로 고정하고 죽는 "자리" 만 바꾼다.
# ---------------------------------------------------------------------------

_SMALL = ScheduleSpec(lanes=4, batch_size=4, two_phase_split=2, compaction_cost_cycles=1)
_SMALL_BRAM = BramSpec(word_tokens=4, n_ports=1, decision_latency_planes=0)


def _step_result(read_live):
    """apply() 가 보는 필드만 채운다."""

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


def _scattered():
    rl = np.zeros((4, 16), dtype=bool)
    rl[0] = rl[1] = True
    rl[2, ::2] = True          # 8개 — 한 칸 걸러
    rl[3, ::4] = True          # 4개 — 네 칸 걸러
    return rl


def _clustered():
    rl = np.zeros((4, 16), dtype=bool)
    rl[0] = rl[1] = True
    rl[2, :8] = True           # 같은 8개인데 앞으로 모여 있다
    rl[3, :4] = True
    return rl


def _numbers(read_live, spec=_SMALL):
    sr = _step_result(read_live)
    out = {}
    for pol in POLICIES:
        r = apply(sr, spec, _SMALL_BRAM, pol)
        out[pol] = (r.cycles, r.reads.words_bram)
    return out


def test_policy_numbers_match_the_hand_calculation():
    """네 정책의 사이클·워드가 손 계산과 같은지 검증."""

    # dense = 4평면 x ceil(16/4) = 16 사이클, 4평면 x 4워드 = 16 워드
    assert _numbers(_scattered()) == {
        "none": (16, 16),
        "batch": (16, 16),
        "compaction": (15, 11),
        "two_phase": (13, 12),
    }


def test_batch_recovers_only_when_termination_is_clustered():
    """batch 가 none 과 갈리는 조건 검증."""

    scattered, clustered = _numbers(_scattered()), _numbers(_clustered())

    # 흩어져 죽으면 묶음마다 생존자가 남아 아무것도 못 건진다
    assert scattered["batch"] == scattered["none"] == (16, 16)

    # 같은 생존 수인데 뭉쳐서 죽으면 묶음이 통째로 빠진다
    assert clustered["batch"] == (11, 11)
    assert clustered["none"] == (16, 16)


def test_only_compaction_reaches_the_word_lower_bound():
    """흩어져 죽었을 때 워드 하한에 닿는 정책 확인."""

    got = _numbers(_scattered())

    # 하한 = 4 + 4 + ceil(8/4) + ceil(4/4)
    assert got["compaction"][1] == 11

    for pol in ("none", "batch", "two_phase"):
        assert got[pol][1] > 11, pol


def test_none_policy_ignores_termination_entirely():
    """none 이 종단을 전혀 쓰지 않는지 검증."""

    # 종단이 실제로 일어나는 데이터로 본다 — 아니면 항등적으로 참이 된다
    r = apply(_step_result(_scattered()), _SMALL, _SMALL_BRAM, "none")

    assert r.reads.ideal_saving == 0.0
    assert r.reads.bram_saving == 0.0
    assert r.cycles == dense_cycles(16, 4, _SMALL)


def test_two_phase_phase_one_reads_everything():
    """1단계가 종단과 무관하게 전체를 읽는지 검증."""

    rl = _scattered()
    words_full = _SMALL_BRAM.n_words(16)

    # 1단계 몫이 split 에 정확히 비례한다. 나머지가 2단계 몫이다.
    phase1 = {split: split * words_full for split in (1, 2, 3, 4)}
    total = {1: 16, 2: 12, 3: 13, 4: 16}

    for split, want in total.items():
        spec = ScheduleSpec(lanes=4, batch_size=4, two_phase_split=split,
                            compaction_cost_cycles=1)
        got = _numbers(rl, spec)["two_phase"][1]

        assert got == want, f"split={split}: {got} != {want}"
        assert got - phase1[split] >= 0

    # split 이 전 평면을 덮으면 2단계가 없어 dense 와 같아진다
    assert total[4] == 4 * words_full


def test_compaction_beats_batch_on_reads():
    """실제 데이터에서 압축이 흩어진 묶음보다 워드가 적은지 검증."""

    p, b, _ = _case(1)
    rb = run_design("exact", p, b, top_k=8, schedule_policy="batch", sched=SCHED, bram=BRAM)
    rc = run_design("exact", p, b, top_k=8, schedule_policy="compaction", sched=SCHED, bram=BRAM)

    # 부등식을 조인다 — 둘이 같으면 압축이 아무 일도 안 한 것이다
    assert rc.reads.words_bram < rb.reads.words_bram


def test_pipeline_efficiency_bounded():
    p, b, _ = _case(4)
    for pol in POLICIES:
        r = run_design("exact", p, b, top_k=8, schedule_policy=pol, sched=SCHED, bram=BRAM)
        assert 0.0 <= r.schedule.pipeline_efficiency <= 1.0 + 1e-9


# ===========================================================================
# 설정 배선 — config/hardware.yaml -> ScheduleSpec / BramSpec
#
# 두 함수 모두 "없으면 기본값" 이고, yaml 값이 dataclass 기본값과 11개 전부
# 같다. 그래서 `spec.lanes == 32` 같은 확인은 파싱이 통째로 실패해도 통과한다.
# 실제로 키 이름을 하나씩 망가뜨려도 저장소 전체 테스트가 그대로 통과했다.
#
# 대신 두 갈래로 본다.
#   1. 기본값과 다른 표지값을 넣은 임시 설정으로 왕복시킨다
#   2. 기본값으로 흘러내리면 ConfigDefaultWarning 이 뜨는지 본다
# ===========================================================================

# 표지값 — 서로 다르고(교차 배선 검출) 기본값과도 다르게(흘러내림 검출) 잡는다.
_SENTINEL = {
    "lanes": 7,
    "batch_size": 11,
    "two_phase_split": 5,
    "compaction_cost_cycles": 9,
    "baseline_pe": 13,
    "baseline_cycles_per_token": 4,
    "mem_overlap": False,
    "word_tokens": 6,
    "word_bits": 17,
    "n_ports": 19,
    "decision_latency_planes": 8,
}

_ALL_WIRING = [("spec", ScheduleSpec, SPEC_WIRING), ("bram", BramSpec, BRAM_WIRING)]

# yaml 에 있어도 파라미터가 아닌 키
_NOT_A_PARAMETER = {"source"}

# yaml 에 있는데 이 모듈이 읽지 않는 키. 늘어나면 여기에 이유를 적을 것.
_UNREAD_ON_PURPOSE = {
    ("datapath", "adder_tree_width"): "RTL 전용 — 소프트웨어 모델에 가산 트리 폭은 안 나온다",
}


def _yaml_scalar(v):
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


def _config_with(tmp_path, values):
    """{점표기: 값} 만 담은 최소 hardware.yaml 을 만든다."""

    cdir = tmp_path / "config"
    cdir.mkdir(parents=True)

    # 나머지 세 파일은 실제 설정을 그대로 쓴다
    for name in CONFIG_FILES:
        if name != "hardware":
            shutil.copyfile(CONFIG_DIR / f"{name}.yaml", cdir / f"{name}.yaml")

    sections: dict[str, dict] = {}
    for dotted, val in values.items():
        _root, sec, key = dotted.split(".")
        sections.setdefault(sec, {})[key] = val

    # meta / resources 는 일부러 뺀다 — 없어도 스케줄 파싱이 살아야 한다
    lines = []
    for sec, kv in sections.items():
        lines.append(f"{sec}:")
        lines += [f"  {k}: {_yaml_scalar(v)}" for k, v in kv.items()]
    (cdir / "hardware.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return cdir


def _sentinel_yaml():
    return {
        dotted: _SENTINEL[f]
        for _w, _c, wiring in _ALL_WIRING
        for f, dotted, _cast in wiring
    }


# --- 이 테스트들 자신의 전제 ------------------------------------------------

def test_sentinels_are_usable_as_evidence():
    """표지값 규약 검증."""

    vals = [repr(v) for v in _SENTINEL.values()]

    # 표지값끼리 겹치지 않음 — 겹치면 교차 배선을 못 잡는다
    assert len(set(vals)) == len(vals)

    for _which, cls, wiring in _ALL_WIRING:
        for fname, _dotted, _cast in wiring:
            # 모든 배선 항목에 표지값이 있음
            assert fname in _SENTINEL, f"{cls.__name__}.{fname}"

            # 표지값 != 기본값 — 같으면 흘러내림을 못 잡는다
            assert _SENTINEL[fname] != getattr(cls(), fname), fname


def test_every_field_is_wired_to_the_config():
    """dataclass 필드 전수가 배선표에 있는지 검증."""

    import dataclasses

    for _which, cls, wiring in _ALL_WIRING:
        wired = {f for f, _d, _c in wiring}

        for f in dataclasses.fields(cls):
            # 설정에서 안 오는 필드 없음 — n_ports 가 그런 상태였다
            assert f.name in wired, f"{cls.__name__}.{f.name} does not come from the config"


# --- 1. 표지값 왕복 ---------------------------------------------------------

def test_config_round_trip_with_sentinel_values(tmp_path):
    """설정 값이 dataclass 까지 도달하는지 검증."""

    cfg = load_config(_config_with(tmp_path, _sentinel_yaml()))

    got = {"spec": spec_from_config(cfg), "bram": bram_from_config(cfg)}

    for which, _cls, wiring in _ALL_WIRING:
        for fname, dotted, _cast in wiring:
            want, have = _SENTINEL[fname], getattr(got[which], fname)

            # 값과 형이 모두 일치 — 표지값이 다 달라 교차 배선도 여기서 걸린다
            assert have == want and type(have) is type(want), (
                f"{dotted} -> {which}.{fname} : {have!r} (expected {want!r})"
            )


def test_config_round_trip_without_pyyaml(tmp_path, monkeypatch):
    """PyYAML 없는 환경의 미니 파서 경로 검증."""

    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "yaml":
            raise ImportError("PyYAML hidden by this test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)

    cfg = load_config(_config_with(tmp_path, _sentinel_yaml()))
    got = {"spec": spec_from_config(cfg), "bram": bram_from_config(cfg)}

    for which, _cls, wiring in _ALL_WIRING:
        for fname, dotted, _cast in wiring:
            # 미니 파서도 같은 값을 낸다
            assert getattr(got[which], fname) == _SENTINEL[fname], f"mini parser: {dotted}"


# --- 2. 기본값으로 흘러내리면 흔적이 남는가 ---------------------------------

def test_real_config_loads_without_a_single_warning():
    """실제 설정 파일의 항목 누락 감시."""

    cfg = load_config()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec_from_config(cfg)
        bram_from_config(cfg)

    missed = [str(w.message) for w in caught if issubclass(w.category, ConfigDefaultWarning)]

    # 기본값으로 흘러내린 항목 없음 — 경고문이 어느 항목인지 알려 준다
    assert not missed, "config entries fell back to defaults:\n  " + "\n  ".join(missed)


def test_a_missing_key_warns_and_names_it(tmp_path):
    """키 누락 시 경고 내용 검증."""

    values = _sentinel_yaml()
    values["hardware.memory.word_tokns"] = values.pop("hardware.memory.word_tokens")
    cfg = load_config(_config_with(tmp_path, values))

    # 빠진 항목의 점표기 경로가 경고문에 있음
    with pytest.warns(ConfigDefaultWarning, match="hardware.memory.word_tokens") as rec:
        bram = bram_from_config(cfg)

    # 기본값으로 흘러내림
    assert bram.word_tokens == BramSpec().word_tokens

    # 오타 후보를 짚어 줌 — 이름만 보면 대체 메시지("keys present in this section: [...]")
    # 에도 그 이름이 들어 있어 제안 경로가 죽어도 통과한다. 문구까지 못 박는다.
    assert "did you mean 'word_tokns'" in str(rec[0].message), str(rec[0].message)

    # 다른 항목은 멀쩡함
    assert bram.n_ports == _SENTINEL["n_ports"]


def test_a_missing_section_warns_once_and_lists_the_fields(tmp_path):
    """섹션 누락 시 경고 내용 검증."""

    values = {d: v for d, v in _sentinel_yaml().items() if not d.startswith("hardware.memory.")}
    cfg = load_config(_config_with(tmp_path, values))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bram = bram_from_config(cfg)

    msgs = [str(w.message) for w in caught if issubclass(w.category, ConfigDefaultWarning)]

    # 섹션 하나에 경고 하나 — 항목마다 떠들지 않는다
    assert len(msgs) == 1, msgs

    # 섹션 이름과 빠진 항목 이름이 경고문에 있음
    assert "hardware.memory" in msgs[0]
    for key in ("word_tokens", "n_bram_ports"):
        assert key in msgs[0], key

    # 전부 기본값
    assert bram == BramSpec()


def test_a_bad_value_raises_and_names_the_path(tmp_path):
    """형 변환 실패는 경고가 아니라 예외인지 검증."""

    values = _sentinel_yaml()

    # 참/거짓이 아닌 값 — bool("false") == True 사고를 막는 자리
    values["hardware.schedule.mem_overlap"] = "maybe"
    with pytest.raises(ValueError, match="hardware.schedule.mem_overlap"):
        spec_from_config(load_config(_config_with(tmp_path / "a", values)))

    # 정수가 아닌 값
    values["hardware.schedule.mem_overlap"] = False
    values["hardware.datapath.lanes"] = "서른둘"
    with pytest.raises(ValueError, match="hardware.datapath.lanes"):
        spec_from_config(load_config(_config_with(tmp_path / "b", values)))


def test_override_typo_lists_the_valid_fields():
    """오버라이드 인자 처리 검증."""

    cfg = load_config()

    # 올바른 이름은 파일 값을 덮어씀
    assert spec_from_config(cfg, lanes=99).lanes == 99
    assert bram_from_config(cfg, word_tokens=99).word_tokens == 99

    # 오타는 조용히 무시되지 않고, 쓸 수 있는 항목을 알려 줌
    with pytest.raises(TypeError, match="word_tokens"):
        bram_from_config(cfg, wordtokens=99)
    with pytest.raises(TypeError, match="lanes"):
        spec_from_config(cfg, lanez=99)


# --- 3. 반대 방향 — yaml 에 있는데 아무도 안 읽는 항목 ----------------------

def test_no_new_dead_parameter_appears():
    """읽히지 않는 설정 항목 감시."""

    cfg = load_config()
    read = {
        tuple(dotted.split(".")[1:])
        for _w, _c, wiring in _ALL_WIRING
        for _f, dotted, _cast in wiring
    }

    dead = set()
    for sec in ("datapath", "schedule", "memory"):
        for key in cfg.get(f"hardware.{sec}", {}) or {}:
            if key not in _NOT_A_PARAMETER and (sec, key) not in read:
                dead.add((sec, key))

    # 죽은 항목은 알려진 것뿐 — n_bram_ports 가 이 상태였다
    assert dead == set(_UNREAD_ON_PURPOSE), (
        f"newly dead: {sorted(dead - set(_UNREAD_ON_PURPOSE))} / "
        f"back in use: {sorted(set(_UNREAD_ON_PURPOSE) - dead)}"
    )

    # 죽은 항목이라도 yaml 주석의 약속(= head_dim)은 지켜야 한다
    assert cfg.get("hardware.datapath.adder_tree_width") == cfg.head_dim


def test_the_one_deliberate_name_mismatch_is_still_there():
    """키 이름과 필드 이름이 다른 항목 고정."""

    mismatched = [
        (fname, dotted)
        for _w, _c, wiring in _ALL_WIRING
        for fname, dotted, _cast in wiring
        if fname != dotted.rpartition(".")[2]
    ]

    # n_bram_ports -> n_ports 하나뿐 — 늘어나면 흘러내림을 알아채기 더 어려워진다
    assert mismatched == [("n_ports", "hardware.memory.n_bram_ports")]


def test_compaction_charges_its_own_cost():
    """★ 압축 오버헤드가 사이클에 실제로 실리는지 — 정책마다 청구 횟수가 다르다.

    compaction 은 평면마다 한 번, two_phase 는 전체에서 한 번 낸다.
    none / batch 는 압축을 안 하므로 청구하지 않는다.

    돌연변이 시험에서 이 항을 0 으로 만들었을 때 핵심 테스트 중
    손 계산 하나만 잡아냈다. 그 하나가 바뀌면 조용해지므로 직접 고정한다.
    """
    sr = _step_result(_scattered())
    n_planes = 4

    def cycles_at(cost, policy):
        return apply(sr, replace(_SMALL, compaction_cost_cycles=cost),
                     _SMALL_BRAM, policy).cycles

    # 비용을 0 -> 5 로 올렸을 때 늘어나는 사이클 = 청구 횟수 x 5
    assert cycles_at(5, "compaction") - cycles_at(0, "compaction") == 5 * n_planes
    assert cycles_at(5, "two_phase") - cycles_at(0, "two_phase") == 5
    assert cycles_at(5, "none") == cycles_at(0, "none")
    assert cycles_at(5, "batch") == cycles_at(0, "batch")

    # 워드 읽기는 압축 비용과 무관하다 — 비용은 시간이지 읽기가 아니다
    for pol in POLICIES:
        a = apply(sr, replace(_SMALL, compaction_cost_cycles=0), _SMALL_BRAM, pol)
        b = apply(sr, replace(_SMALL, compaction_cost_cycles=9), _SMALL_BRAM, pol)
        assert a.reads.words_bram == b.reads.words_bram, pol


# ===========================================================================
# 돌연변이·커버리지 감사에서 나온 구멍 메우기
#
# 위 검사들이 통과하는데도 아래 다섯 가지 변경이 저장소 전체에서 아무에게도
# 안 걸렸다. 원인은 검사가 약해서가 아니라 **픽스처가 그 축을 못 건드려서**다.
#   _SMALL 은 batch_size(4) == word_tokens(4) 라 묶음 하나가 곧 워드 하나이고,
#   _scattered() 는 평면 0·1 이 전부 생존이라 1단계 안에서 종단이 안 일어난다.
# 그래서 필요한 축을 건드리는 픽스처를 따로 만든다.
# ===========================================================================

SR_SMALL = _step_result(_scattered())


def _clustered_in_batch():
    """묶음(8) 안에서 생존이 앞쪽 워드(4)에 뭉친 read_live."""
    rl = np.zeros((4, 16), dtype=bool)
    rl[0] = rl[1] = True
    rl[2, 0:4] = True
    rl[2, 8:12] = True
    rl[3, 0:4] = True
    return rl


def _dies_inside_phase_one():
    """1단계(평면 0..1) 안에서 이미 종단이 일어나는 read_live."""
    rl = np.zeros((4, 16), dtype=bool)
    rl[0] = True
    rl[1, :8] = True          # 평면 1 에서 절반이 죽는다
    rl[2, :4] = True
    rl[3, :2] = True
    return rl


def test_batch_drags_the_whole_batch_through_memory():
    """★ 묶음이 살아 있으면 그 안의 죽은 워드까지 읽는지 — 회계에 실리는지 검증.

    _SMALL 은 batch_size == word_tokens 라 묶음과 워드가 1:1 이고, 그래서
    '묶음 끌기'를 없애도 숫자가 안 변한다. 폭을 어긋나게 해야 드러난다.
    """
    spec = replace(_SMALL, batch_size=8)          # 묶음 8 != 워드 4
    r = apply(_step_result(_clustered_in_batch()), spec, _SMALL_BRAM, "batch")

    # 살아 있는 묶음은 통째로 읽는다. 생존이 앞 워드에만 있어도 뒤 워드를 낸다.
    assert r.reads.words_bram == 14, r.reads.words_bram

    # 실제로 읽는 양(끌기 포함)이 압축 하한보다 커야 batch 가 batch 다
    assert r.reads.words_bram > r.reads.words_compacted


def test_two_phase_phase_one_ignores_termination_in_the_word_count():
    """★ 1단계는 종단과 무관하게 전체를 읽는다 — 워드 회계로 확인.

    test_two_phase_phase_one_reads_everything 은 1단계 안에서 종단이 없는
    픽스처를 써서, 이름이 주장하는 바를 정작 검증하지 못한다.
    """
    spec = replace(_SMALL, two_phase_split=2)
    rl = _dies_inside_phase_one()
    r = apply(_step_result(rl), spec, _SMALL_BRAM, "two_phase")

    words_full = _SMALL_BRAM.n_words(16)
    compacted_phase1 = sum(word_reads_compacted(int(rl[t].sum()), _SMALL_BRAM.word_tokens)
                           for t in range(2))

    # 1단계가 종단을 반영하면 이 값이 더 작아진다 — 그 차이가 존재해야 검사가 산다
    assert compacted_phase1 < 2 * words_full, "fixture does not exercise phase-1 termination"
    assert r.reads.words_bram == 10, r.reads.words_bram


def test_an_unknown_policy_raises_instead_of_falling_into_two_phase():
    """★ 정책 이름 오타가 조용히 two_phase 로 흘러들지 않는지 검증.

    if/elif 사슬에 최종 else 방어가 없어서, 이름 검증을 빼면 인식 못 한 문자열이
    전부 two_phase 가지로 간다. 예외가 아니라 '틀린 수치 + 없는 라벨'이 나오고
    그 라벨이 as_dict() 를 타고 실험 CSV 에 그대로 실린다.
    """
    with pytest.raises(KeyError, match="compation"):
        apply(SR_SMALL, _SMALL, _SMALL_BRAM, "compation")

    # 빈 문자열·대소문자도 마찬가지
    for bad in ("", "COMPACTION", "two-phase"):
        with pytest.raises(KeyError):
            apply(SR_SMALL, _SMALL, _SMALL_BRAM, bad)


def test_split_beyond_the_plane_count_is_clamped():
    """★ split 이 평면 수를 넘어도 물리적으로 불가능한 회계가 나오지 않는지 검증."""
    for split in (5, 99):
        r = apply(SR_SMALL, replace(_SMALL, two_phase_split=split), _SMALL_BRAM, "two_phase")

        # 읽은 워드가 dense 를 넘을 수는 없다
        assert r.reads.words_bram <= r.reads.words_dense, (split, r.reads.words_bram)
        assert r.reads.reads_ideal <= r.reads.reads_dense, split

        # split 이 전 평면을 덮으면 dense 와 같아진다
        assert r.reads.words_bram == r.reads.words_dense

    # 0 이하도 1 로 잡힌다 — 음수 회계가 나오면 안 된다
    r0 = apply(SR_SMALL, replace(_SMALL, two_phase_split=0), _SMALL_BRAM, "two_phase")
    assert r0.reads.reads_ideal > 0


def test_two_phase_rejects_non_monotone_read_live():
    """★ 압축 집합 밖의 토큰을 나중에 읽는 입력을 잡아내는지 검증.

    two_phase 는 압축을 한 번만 하므로, 압축 시점에 죽어 있던 토큰이 뒤 평면에서
    되살아나면 그 토큰은 압축 배열에 자리가 없다. 조용히 세면 회계가 틀어진다.
    """
    rl = np.zeros((4, 16), dtype=bool)
    rl[0] = rl[1] = True
    rl[2, :4] = True          # 압축 집합 = 토큰 0..3
    rl[3, 8:12] = True        # 그 밖의 토큰을 읽는다 — 단조가 아니다

    with pytest.raises(ValueError, match="not monotone"):
        apply(_step_result(rl), replace(_SMALL, two_phase_split=2), _SMALL_BRAM, "two_phase")
