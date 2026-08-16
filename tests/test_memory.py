"""★ BRAM 워드 단위 읽기 회계 검증 ★

이론 절감과 실현 절감을 분리하는 장치가 제대로 동작하는지 확인한다.
이게 틀리면 논문의 핵심 수치가 틀린다.
"""

import numpy as np

from src.memory import (
    BramSpec,
    account_step,
    bitplane_bram_bytes,
    kv_cache_bytes,
    word_reads_compacted,
    word_reads_scattered,
)


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
    assert acc.ideal_saving > 0.5, "이론적으로는 절반 넘게 줄어야 한다"
    assert acc.bram_saving < 0.05, "그런데 흩어져 있으면 워드 단위로는 거의 안 준다"
    assert acc.compacted_saving > 0.5, "압축하면 회복되어야 한다"
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
