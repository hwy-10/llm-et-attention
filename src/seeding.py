"""난수 시드 관리.

같은 시드로 여러 사람이 같은 결과를 얻어야 "정확도가 유지된다"는 주장이 재현된다.
모든 난수는 여기서 만든 Generator 를 통해서만 뽑는다.

★ 2026-08-28 정리 ★
이전 판에는 아날로그 CIM 을 전제로 한 `instance_rng`(매크로 인스턴스 고정 패턴,
INL·커패시터 미스매치)와 `trial_rng`(열잡음 몬테카를로)가 있었다. 이 프로젝트는
**디지털 회로만** 다루므로 범위 밖이고, 저장소 어디서도 호출되지 않는 죽은 코드였다.
발표에서 "아날로그도 하시나요"라는 오해를 부르기도 한다.

지금 쓰는 난수는 **합성 Q/K 생성** 하나뿐이다.
"""

from __future__ import annotations

import numpy as np

# spawn_key 로 흐름을 분리해 둔다. 나중에 다른 종류의 난수가 필요해지면
# 같은 seed 를 써도 서로 간섭하지 않는다.
_DATA_STREAM = 2


def data_rng(seed: int = 0) -> np.random.Generator:
    """합성 Q/K 텐서 생성용 RNG.

    같은 seed 면 언제 어디서 불러도 같은 수열이 나온다.
    """
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(_DATA_STREAM,)))
