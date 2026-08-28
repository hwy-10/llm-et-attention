"""★ BRAM 워드 단위 읽기 회계 검증 ★

이론 절감과 실현 절감을 분리하는 장치가 제대로 동작하는지 확인한다.
이게 틀리면 논문의 핵심 수치가 틀린다.
"""

import warnings

import numpy as np
import pytest

from src.config import ConfigDefaultWarning, load_config
from src.memory import (
    BramSpec,
    account_step,
    bitplane_bram_bytes,
    capacity_report,
    kv_cache_bytes,
    word_reads_compacted,
    word_reads_scattered,
)
from src.schedule import bram_from_config


def test_word_reads_scattered_hand_calc():
    # 워드폭 4, 토큰 12개 -> 워드 3개
    live = np.array([1, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 1], dtype=bool)
    assert word_reads_scattered(live, 4) == 2      # 0번, 2번 워드
    live2 = np.zeros(12, dtype=bool)
    assert word_reads_scattered(live2, 4) == 0
    assert word_reads_scattered(np.ones(12, dtype=bool), 4) == 3


def test_word_reads_padding():
    """토큰 수가 워드폭의 배수가 아니어도 정확해야 한다."""
    live = np.zeros(10, dtype=bool)
    live[9] = True
    assert word_reads_scattered(live, 4) == 1      # 마지막(3번) 워드만


def test_word_reads_compacted():
    assert word_reads_compacted(0, 32) == 0
    assert word_reads_compacted(1, 32) == 1
    assert word_reads_compacted(32, 32) == 1
    assert word_reads_compacted(33, 32) == 2


def test_word_tokens_one_means_ideal_equals_realized():
    """★ 워드폭 1이면 이론 절감이 그대로 실현되어야 한다 ★"""
    rng = np.random.default_rng(0)
    rl = rng.random((8, 100)) < 0.6
    acc = account_step(rl, BramSpec(word_tokens=1))
    assert acc.words_bram == acc.reads_ideal
    assert abs(acc.bram_saving - acc.ideal_saving) < 1e-12
    assert abs(acc.realization_ratio - 1.0) < 1e-12


def test_wide_words_destroy_scattered_savings():
    """★ 이 프로젝트의 핵심 함정 ★

    살아있는 토큰이 흩어져 있으면 넓은 워드에서는 절감이 거의 실현되지 않는다.
    반면 압축하면 회복된다.
    """
    n_tokens, n_planes = 1024, 8
    rng = np.random.default_rng(1)
    # 각 평면에서 40% 만 살아있되, 위치는 무작위로 흩어져 있다
    rl = rng.random((n_planes, n_tokens)) < 0.4

    acc = account_step(rl, BramSpec(word_tokens=64))
    assert acc.ideal_saving > 0.5, "in theory more than half should be saved"
    assert acc.bram_saving < 0.05, "but scattered survivors realize almost nothing per word"
    assert acc.compacted_saving > 0.5, "compaction must recover it"
    assert acc.realization_ratio < 0.1


def test_contiguous_live_realizes_savings():
    """살아있는 토큰이 앞쪽에 몰려 있으면 워드 단위로도 절감된다."""
    n_tokens, n_planes = 256, 8
    rl = np.zeros((n_planes, n_tokens), dtype=bool)
    rl[:, :100] = True
    acc = account_step(rl, BramSpec(word_tokens=64))
    assert acc.realization_ratio > 0.8


def test_account_accumulation():
    rl = np.ones((8, 64), dtype=bool)
    spec = BramSpec(word_tokens=32)
    a, b = account_step(rl, spec), account_step(rl, spec)
    a += b
    assert a.reads_ideal == 2 * 8 * 64
    assert a.words_dense == 2 * 8 * 2


def test_kv_cache_size_matches_guide():
    """가이드 3.5절: Llama 3.2 1B, 토큰 1개당 32 KB, 512 토큰이면 16 MB."""
    per_token = kv_cache_bytes(16, 8, 64, 1, 2)
    assert per_token == 32768
    assert kv_cache_bytes(16, 8, 64, 512, 2) == 16 * 1024 * 1024


def test_bitplane_capacity():
    """1층 1헤드, 512 토큰, head_dim 64, INT8 -> 32 KB"""
    assert bitplane_bram_bytes(512, 64, 8) == 512 * 64


# ===========================================================================
# 워드 단위 읽기 회계 — 손 계산 대조와 방어선
#
# 이 파일의 수치가 틀리면 두 팀이 아무리 검증해도 최종 절감률이 틀린다.
# 그래서 무작위 성질 검사 대신 손으로 셀 수 있는 값과 맞춘다.
# ===========================================================================


def test_word_reads_scattered_by_hand():
    """워드폭 4에서 흩어진 생존이 몇 워드인지 손 계산과 대조."""

    cases = [
        # 생존 마스크                          기대  이유
        ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 2),  # 워드 0 과 2 에만
        ([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0], 2),  # 워드 0 꽉참, 워드 2 하나
        ([0] * 12, 0),                              # 전멸
        ([1] * 12, 3),                              # 전부 생존
        ([0, 0, 0, 1, 1, 0, 0, 0], 2),              # 워드 경계를 걸친 두 개
    ]

    for row, want in cases:
        got = word_reads_scattered(np.array(row, dtype=bool), 4)
        assert got == want, f"{row}: {got} != {want}"


def test_padding_when_tokens_do_not_fill_the_last_word():
    """토큰 수가 워드폭의 배수가 아닐 때 처리 검증."""

    # 전부 생존이면 ceil(n / word_tokens) 와 같아야 한다
    for n, wt in ((10, 4), (1, 4), (5, 8), (0, 4), (12, 5)):
        assert word_reads_scattered(np.ones(n, dtype=bool), wt) == -(-n // wt)

    # 마지막 토큰 하나만 생존 — 패딩 자리가 유령 워드를 만들면 안 된다
    last = np.zeros(10, dtype=bool)
    last[9] = True
    assert word_reads_scattered(last, 4) == 1

    # 패딩은 평면마다 따로 붙는다. 평면 경계를 넘어 묶이면 값이 달라진다.
    rl = np.zeros((2, 6), dtype=bool)
    rl[0, 5] = True
    rl[1, 0] = True
    assert sum(word_reads_scattered(rl[t], 4) for t in range(2)) == 2


def test_scattered_never_beats_compacted():
    """압축이 흩어진 배치보다 항상 적거나 같은지 검증."""

    rng = np.random.default_rng(0)
    gap = 0

    for _ in range(500):
        n, wt = int(rng.integers(1, 65)), int(rng.integers(1, 17))
        live = rng.random(n) < rng.random()
        s = word_reads_scattered(live, wt)
        c = word_reads_compacted(int(live.sum()), wt)

        assert c <= s, f"n={n} wt={wt}: compacted {c} > scattered {s}"
        gap = max(gap, s - c)

    # 격차가 0 만 나오면 이 테스트가 아무것도 안 본 것이다
    assert gap > 0


def test_n_words_rounds_up():
    """마지막 조각도 워드 하나를 차지하는지 검증."""

    spec = BramSpec(word_tokens=4)

    # 나누어떨어지지 않는 값이 판별점이다 — 8, 256 만 보면 올림인지 알 수 없다
    assert spec.n_words(9) == 3
    assert spec.n_words(1) == 1

    assert spec.n_words(8) == 2
    assert spec.n_words(0) == 0


def test_savings_are_ratios_of_sums_not_sums_of_ratios():
    """여러 스텝을 합칠 때 절감률이 어떻게 나오는지 검증."""

    rl_big = np.zeros((2, 8), dtype=bool)
    rl_big[0] = True                      # 워드 2개
    rl_small = np.zeros((2, 8), dtype=bool)
    rl_small[0, 0] = True                 # 워드 1개

    spec = BramSpec(word_tokens=4)
    total = account_step(rl_big, spec)
    total += account_step(rl_small, spec)

    # 합계에서 다시 계산한다 — 스텝별 비율을 평균 내면 큰 스텝이 묻힌다
    assert total.words_bram == 2 + 1
    assert total.words_dense == 4 + 4
    assert total.bram_saving == 1.0 - 3 / 8


def test_a_bad_word_width_raises_and_names_it():
    """워드폭이 0 이하일 때 조용히 넘어가지 않는지 검증."""

    for wt in (0, -4):
        with pytest.raises(ValueError, match="word_tokens"):
            word_reads_compacted(8, wt)
        with pytest.raises(ValueError, match="word_tokens"):
            BramSpec(word_tokens=wt).n_words(8)

    with pytest.raises(ValueError, match="word_tokens"):
        word_reads_scattered(np.ones(8, dtype=bool), 0)

    # 음수 토큰 수도 0 으로 삼키지 않는다
    with pytest.raises(ValueError, match="negative"):
        BramSpec().n_words(-5)


def test_passing_the_whole_array_raises():
    """★ 평면 하나가 아니라 2차원을 통째로 넘기면 막는지 검증.

    막지 않으면 평면 경계를 넘어 한 워드로 묶여 조용히 다른 값이 나온다.
    """
    rl = np.zeros((2, 6), dtype=bool)
    rl[0, 5] = True
    rl[1, 0] = True

    with pytest.raises(ValueError, match="1-D"):
        word_reads_scattered(rl, 4)


def test_word_counts_are_exact_for_huge_inputs():
    """정수 나눗셈이므로 float 정밀도 한계를 넘어도 어긋나지 않는다."""

    huge = 2**53 + 1

    assert word_reads_compacted(huge, 1) == huge
    assert BramSpec(word_tokens=1).n_words(huge) == huge


def test_account_step_output_is_words_not_tokens():
    """account_step 의 출력 단위가 BRAM 워드 수인지 검증."""

    # 평면 4 x 토큰 8, 워드폭 4 -> 평면당 2워드
    rl = np.zeros((4, 8), dtype=bool)
    rl[0] = True                      # 8토큰 = 2워드
    rl[1, [0, 7]] = True              # 양 끝    = 2워드
    rl[2, [0, 1]] = True              # 앞쪽만  = 1워드
    acc = account_step(rl, BramSpec(word_tokens=4, word_bits=32))

    # 토큰 단위와 워드 단위가 서로 다른 값이어야 한다 — 같으면 단위가 섞인 것이다
    assert acc.reads_ideal == 12                     # 살아있는 (평면, 토큰) 쌍
    assert acc.words_bram == 5                       # 워드 수
    assert acc.reads_ideal != acc.words_bram

    assert acc.words_dense == 4 * 2
    assert acc.words_compacted == 2 + 1 + 1 + 0
    assert acc.bits_read == 5 * 32


def test_word_width_one_makes_words_equal_tokens():
    """워드폭 1 에서만 두 단위가 같아지는지 검증 — 단위 혼동의 기준선."""

    rl = np.zeros((4, 8), dtype=bool)
    rl[0] = True
    rl[1, [0, 7]] = True
    acc = account_step(rl, BramSpec(word_tokens=1))

    assert acc.words_bram == acc.reads_ideal
    assert acc.words_dense == acc.reads_dense
    assert acc.bram_saving == acc.ideal_saving


def test_the_shipped_config_keeps_bits_read_meaningful():
    """워드가 담는 토큰 비트를 실제로 실을 수 있는 폭인지 감시."""

    bram = bram_from_config(load_config())

    # 토큰 하나가 평면당 1비트다. 이보다 좁으면 bits_read 가 말이 안 된다.
    assert bram.word_bits >= bram.word_tokens, (
        f"word_bits {bram.word_bits} < word_tokens {bram.word_tokens}"
    )


def test_capacity_numbers_match_the_guide():
    """가이드 3.5절의 용량 수치를 재현하는지 검증."""

    cfg = load_config()
    m = cfg.get("model.model", {})
    n_layers, n_heads = int(m["n_layers"]), int(m["n_kv_heads"])
    head_dim, dtype_bytes = int(m["head_dim"]), int(m.get("kv_dtype_bytes", 2))

    # K+V 합쳐 토큰당 원소 수 = 2 x 층 x 헤드 x 차원
    assert 2 * n_layers * n_heads * head_dim == 16_384

    # 문맥 512 에서 16 MiB
    assert kv_cache_bytes(n_layers, n_heads, head_dim, 512, dtype_bytes) == 16 * 2**20

    # 비트평면 K 1층 1헤드 512토큰 = 32 KB
    assert bitplane_bram_bytes(512, 64, 8) == 32 * 1024

    # 온칩 비트평면은 외부 KV 캐시보다 훨씬 작다 — 이게 설계의 전제다
    rep = capacity_report(cfg, seq_len=512)
    assert rep["k_bitplane_all_heads_bytes"] < rep["kv_cache_external_bytes"]

    # KV 캐시는 n_kv_heads 로 센다. n_heads(어텐션 헤드) 를 쓰면 4배로 부푼다.
    assert int(m["n_heads"]) != n_heads, "이 검사가 기대는 전제가 깨졌다"
    assert rep["k_bitplane_all_heads_bytes"] == (
        rep["k_bitplane_one_head_bytes"] * n_layers * n_heads
    )
    assert rep["kv_cache_external_bytes"] == kv_cache_bytes(
        n_layers, n_heads, head_dim, 512, dtype_bytes
    )


def test_capacity_report_warns_when_the_model_section_is_missing():
    """모델 치수를 못 읽으면 흔적이 남는지 검증.

    네 항목 전부 yaml 값과 기본값이 같아 숫자로는 알아챌 수 없다.
    """
    cfg = load_config()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        good = capacity_report(cfg, seq_len=512)
    assert not [w for w in caught if issubclass(w.category, ConfigDefaultWarning)]

    cfg.model.pop("model")
    with pytest.warns(ConfigDefaultWarning, match="model.model"):
        blind = capacity_report(cfg, seq_len=512)

    # 경고가 유일한 단서다 — 값은 똑같이 나온다
    assert blind == good


def test_readme_word_width_table_is_reproducible():
    """★ README 워드폭 표를 memory.py 로 직접 재현.

    워드폭이 넓을수록 흩어진 생존의 실현 절감이 무너지는지 본다.
    """
    # 평면 8 x 토큰 256, 뒤쪽 평면에서 흩어져 죽는다
    rng = np.random.default_rng(0)
    rl = np.zeros((8, 256), dtype=bool)
    order = rng.permutation(256)
    for t in range(8):
        n_live = 256 if t < 5 else max(16, int(256 * 0.3 ** (t - 4)))
        rl[t, order[:n_live]] = True

    seen = []
    for wt in (1, 8, 32, 64):
        acc = account_step(rl, BramSpec(word_tokens=wt, word_bits=max(32, wt)))
        seen.append(acc.realization_ratio)

    # 워드폭 1 에서는 이론 절감이 그대로 실현된다
    assert seen[0] == pytest.approx(1.0)

    # 워드폭이 넓어질수록 실현률이 단조 감소한다
    for a, b in zip(seen, seen[1:]):
        assert b <= a + 1e-9, seen

    # 워드폭 64 에서는 거의 아무것도 실현되지 않는다
    assert seen[-1] < 0.2, seen
