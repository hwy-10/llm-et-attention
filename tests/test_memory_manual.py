"""BRAM 읽기 회계 수동 예제 검증."""

import numpy as np

from src.memory import BramSpec, account_step


def test_account_step_manual_example():
    """작은 수동 입력으로 이상적 읽기량과 BRAM 워드 읽기량을 검증한다."""

    # 평면 4개, 토큰 8개
    # 한 BRAM 워드에는 토큰 4개의 같은 비트평면이 저장된다.
    read_live = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],  # 평면 0: 워드 2개
            [1, 0, 0, 0, 0, 0, 0, 1],  # 평면 1: 워드 2개
            [1, 1, 0, 0, 0, 0, 0, 0],  # 평면 2: 워드 1개
            [0, 0, 0, 0, 0, 0, 0, 0],  # 평면 3: 워드 0개
        ],
        dtype=bool,
    )

    spec = BramSpec(
        word_tokens=4,
        word_bits=32,
    )

    result = account_step(read_live, spec)

    # 살아 있는 (평면, 토큰) 쌍:
    # 8 + 2 + 2 + 0 = 12
    assert result.reads_ideal == 12

    # 종단이 없을 경우:
    # 평면 4개 × 토큰 8개 = 32
    assert result.reads_dense == 32

    # 실제 BRAM 워드 읽기:
    # 평면별 2 + 2 + 1 + 0 = 5워드
    assert result.words_bram == 5

    # 종단이 없을 경우:
    # 평면 4개 × 평면당 2워드 = 8워드
    assert result.words_dense == 8

    # 생존 토큰을 평면별로 압축한 경우:
    # ceil(8/4) + ceil(2/4) + ceil(2/4) + 0
    # = 2 + 1 + 1 + 0 = 4워드
    assert result.words_compacted == 4

    # 실제 읽은 비트:
    # 5워드 × 32비트 = 160비트
    assert result.bits_read == 160

    assert result.ideal_saving == 0.625
    assert result.bram_saving == 0.375
    assert result.compacted_saving == 0.5
    assert result.realization_ratio == 0.6