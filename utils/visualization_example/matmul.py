"""행렬 곱을 "어느 행과 어느 열이 어느 칸을 만드는가" 단위로 분해한다.

이 모듈이 계산의 전부다. 브라우저는 여기서 나온 값을 **그리기만** 하고
곱셈·덧셈을 다시 하지 않는다. 그래야 화면에 보이는 숫자가 파이썬이 실제로
계산한 값이라는 것이 보장된다 — 두 곳에서 따로 계산하면 둘이 어긋나도
아무도 모른다.

의존성
------
저장소 규약대로 **numpy 없이도 동작**한다. 곱셈은 언제나 아래의 삼중 루프가
하고, numpy 가 있으면 그 결과를 ``A @ B`` 와 대조해 교차 검증한다.
없으면 검증만 건너뛴다 (``backend_name()`` 이 어느 쪽인지 알려 준다).

교차 검증을 넣는 이유는 이 모듈이 "곱셈을 빠르게 하는 코드"가 아니라
**설명용 분해**이기 때문이다. 분해가 실제 행렬 곱과 다르면 교보재로서
가치가 없으므로, 매 호출마다 확인한다.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

try:  # 선택 의존성 — 교차 검증에만 쓴다
    import numpy as _np
except ImportError:  # pragma: no cover - numpy 는 코어 의존성이라 보통 있다
    _np = None

__all__ = [
    "ABS_MIN_DIM",
    "MAX_DIM",
    "MIN_DIM",
    "backend_name",
    "breakdown",
    "multiply",
    "random_matrix",
    "validate_pair",
]

ABS_MIN_DIM = 1  # 계산이 성립하는 최소 — 1 x d 쿼리 벡터가 여기 해당한다
MIN_DIM = 2      # 화면(스테퍼)의 하한. 1x1 격자는 그려 봐야 설명할 것이 없다
MAX_DIM = 12     # 화면에 다 들어가고 손으로 검산 가능한 상한

Matrix = list[list[int]]


def backend_name() -> str:
    """교차 검증에 쓸 수 있는 백엔드 이름."""
    return "numpy" if _np is not None else "pure-python"


# ---------------------------------------------------------------------------
# 생성 · 검사
# ---------------------------------------------------------------------------

def random_matrix(
    rows: int,
    cols: int,
    lo: int = -9,
    hi: int = 9,
    seed: int | None = None,
    exclude_zero: bool = True,
) -> Matrix:
    """손으로 검산 가능한 크기의 정수 행렬을 만든다.

    ``exclude_zero`` 는 기본값이 True 다. 0이 섞이면 "이 항은 왜 사라졌지"를
    설명하느라 정작 보여 주려는 것이 묻힌다.
    """
    _check_dim(rows, "rows")
    _check_dim(cols, "cols")
    if lo > hi:
        raise ValueError(f"lo({lo}) is greater than hi({hi})")
    if exclude_zero and lo <= 0 <= hi and lo == hi == 0:
        raise ValueError("exclude_zero=True but 0 is the only value in range")

    rng = random.Random(seed)
    out: Matrix = []
    for _ in range(rows):
        row: list[int] = []
        for _ in range(cols):
            v = rng.randint(lo, hi)
            while exclude_zero and v == 0:
                v = rng.randint(lo, hi)
            row.append(v)
        out.append(row)
    return out


def _check_dim(n: Any, name: str, lo: int = MIN_DIM) -> int:
    """차원 하나를 검사한다.

    ``lo`` 가 둘로 갈리는 이유가 있다. :data:`MIN_DIM` 은 **화면 제약**이다 —
    1×1 격자를 그려 봐야 설명할 것이 없어서 스테퍼를 2 부터 시작시킨다.
    반면 계산은 1행·1열도 완전히 정당하고, 하필 그것이 이 프로젝트가 다루는
    모양이다 (디코드 스텝의 쿼리는 1 × d_head 한 줄이다). 그래서 들어온 값을
    검사할 때는 :data:`ABS_MIN_DIM` 을 쓴다.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"{name} must be an integer: {n!r}")
    if not (lo <= n <= MAX_DIM):
        raise ValueError(f"{name}={n} is outside the allowed range {lo}..{MAX_DIM}")
    return n


def _as_matrix(m: Any, name: str) -> Matrix:
    """중첩 리스트를 정수 행렬로 정규화한다. 모양이 어긋나면 바로 거절한다."""
    if not isinstance(m, Sequence) or isinstance(m, (str, bytes)):
        raise TypeError(f"{name} must be a 2-D list")
    rows = list(m)
    if not rows:
        raise ValueError(f"{name} is empty")

    out: Matrix = []
    width: int | None = None
    for r, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(f"{name}[{r}] is not a list")
        vals = [_as_int(v, f"{name}[{r}][{c}]") for c, v in enumerate(row)]
        if width is None:
            width = len(vals)
        elif len(vals) != width:
            raise ValueError(
                f"{name} has ragged rows: {width} vs {len(vals)} (row {r})"
            )
        out.append(vals)

    _check_dim(len(out), f"{name} 행 수", ABS_MIN_DIM)
    _check_dim(width or 0, f"{name} 열 수", ABS_MIN_DIM)
    return out


def _as_int(v: Any, where: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{where} is not a number: {v!r}")
    if isinstance(v, float) and not v.is_integer():
        raise ValueError(f"{where} must be an integer: {v!r}")
    return int(v)


def validate_pair(A: Any, B: Any) -> tuple[Matrix, Matrix]:
    """A·B 가 성립하는지 확인하고 정규화한 두 행렬을 돌려준다.

    성립 조건은 하나뿐이다 — **A의 열 수 == B의 행 수**. 이 공통 차원이
    분해에서 k 로 나타난다.
    """
    a = _as_matrix(A, "A")
    b = _as_matrix(B, "B")
    if len(a[0]) != len(b):
        raise ValueError(
            f"cannot multiply: A has {len(a[0])} columns but B has {len(b)} rows"
        )
    return a, b


# ---------------------------------------------------------------------------
# 곱셈
# ---------------------------------------------------------------------------

def multiply(A: Any, B: Any, *, cross_check: bool = True) -> Matrix:
    """C = A · B.

    화면에 그려지는 분해와 **같은 순서로** 삼중 루프를 돈다. numpy 가 있고
    ``cross_check`` 가 True 면 ``A @ B`` 와 대조해 어긋나면 예외를 던진다.
    """
    a, b = validate_pair(A, B)
    m, n, p = len(a), len(b), len(b[0])

    c: Matrix = [[0] * p for _ in range(m)]
    for i in range(m):
        row_a = a[i]
        for j in range(p):
            acc = 0
            for k in range(n):
                acc += row_a[k] * b[k][j]
            c[i][j] = acc

    if cross_check and _np is not None:
        ref = (_np.asarray(a, dtype=object) @ _np.asarray(b, dtype=object)).tolist()
        if ref != c:
            raise AssertionError(
                "manual triple loop disagrees with numpy A @ B.\n"
                f"  manual = {c}\n  numpy  = {ref}"
            )
    return c


def breakdown(A: Any, B: Any, i: int, j: int) -> dict[str, Any]:
    """C[i][j] 한 칸이 만들어지는 과정을 항 단위로 펼친다.

    돌려주는 것은 JSON 으로 그대로 나갈 수 있는 dict 다::

        {
          "i": 1, "j": 2,
          "row":  [3, -7, 1, 4],      # A 의 i 행
          "col":  [2, 5, -1, 8],      # B 의 j 열
          "terms": [{"k":0, "a":3, "b":2, "prod":6, "cumsum":6}, ...],
          "total": 2,
          "cross_checked": True,
        }

    ``cumsum`` 은 누산기(accumulator)가 k 단계까지 들고 있는 값이다. 곱이
    한 번에 나오는 값이 아니라 **순차 누산**이라는 것을 보이려고 넣었다.
    """
    a, b = validate_pair(A, B)
    m, n, p = len(a), len(b), len(b[0])
    if not (0 <= i < m):
        raise IndexError(f"row index i={i} is outside 0..{m - 1}")
    if not (0 <= j < p):
        raise IndexError(f"column index j={j} is outside 0..{p - 1}")

    row = list(a[i])
    col = [b[k][j] for k in range(n)]

    terms: list[dict[str, int]] = []
    acc = 0
    for k in range(n):
        prod = row[k] * col[k]
        acc += prod
        terms.append({"k": k, "a": row[k], "b": col[k], "prod": prod, "cumsum": acc})

    checked = False
    if _np is not None:
        ref = int(_np.dot(_np.asarray(row, dtype=object), _np.asarray(col, dtype=object)))
        if ref != acc:
            raise AssertionError(
                f"term-by-term sum ({acc}) disagrees with numpy dot ({ref}) at i={i}, j={j}"
            )
        checked = True

    return {
        "i": i,
        "j": j,
        "row": row,
        "col": col,
        "terms": terms,
        "total": acc,
        "cross_checked": checked,
    }
