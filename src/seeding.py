"""난수 시드 관리.

아날로그 오차는 난수다. 시드를 통제하지 않으면 "정확도가 유지된다"는 주장이
재현되지 않는다. 모든 난수는 여기서 만든 Generator 를 통해서만 뽑는다.

두 종류의 난수를 구분한다:
  * instance 난수 — 매크로 인스턴스 고정 패턴 (INL, 커패시터 미스매치).
                    같은 칩이면 매번 같아야 하므로 instance_seed 로 고정.
  * trial 난수    — 샘플마다 달라지는 열잡음. 몬테카를로 반복마다 바뀐다.
"""

from __future__ import annotations

import numpy as np

_DEFAULT_INSTANCE_SEED = 1234


def trial_rng(seed: int) -> np.random.Generator:
    """몬테카를로 반복용 RNG (샘플마다 달라지는 잡음)."""
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(0,)))


def instance_rng(instance_seed: int = _DEFAULT_INSTANCE_SEED) -> np.random.Generator:
    """매크로 인스턴스 고정 패턴용 RNG (INL, 커패시터 미스매치)."""
    return np.random.default_rng(np.random.SeedSequence(entropy=instance_seed, spawn_key=(1,)))


def data_rng(seed: int = 0) -> np.random.Generator:
    """합성 Q/K 텐서 생성용 RNG."""
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(2,)))
