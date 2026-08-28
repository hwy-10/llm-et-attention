"""pytest 없이도 테스트를 돌리는 러너.

    python tests/run_tests.py            # 전체
    python tests/run_tests.py test_bounds

pytest 가 설치돼 있으면 그쪽으로 위임하고, 없으면 최소 셔임을 주입해 직접 실행한다.
저장소를 클론하자마자 검증이 가능하게 하려는 장치다 — README 의 "의존성은 numpy
하나" 가 이 경로로 지켜진다.

셔임이 흉내내는 것: approx · raises · warns · fixture · mark.parametrize,
그리고 tmp_path / monkeypatch 픽스처. **여기 없는 기능을 검사에서 쓰면 CI 의
무-pytest 잡만 깨진다** — 로컬에는 pytest 가 있어 통과하기 때문에 안 보인다.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# pytest 최소 셔임
# ---------------------------------------------------------------------------
class _Approx:
    def __init__(self, expected, rel=1e-6, abs_=1e-12):
        self.expected, self.rel, self.abs = expected, rel, abs_

    def __eq__(self, other):
        return abs(other - self.expected) <= max(
            self.abs, self.rel * abs(self.expected)
        )

    def __repr__(self):
        return f"approx({self.expected!r}, rel={self.rel})"


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match
        self.value = None          # pytest 는 잡은 예외를 .value 로 준다

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"{self.exc.__name__} was not raised")
        if not issubclass(et, self.exc):
            return False
        self.value = ev
        # pytest 의 match= 는 부분문자열이 아니라 re.search 다.
        # 부분문자열로 두면 r"\[0, 255\]" 같은 패턴이 pytest 에서는 통과하고
        # 여기서만 실패한다 — 러너에 따라 결과가 갈리면 어느 쪽도 못 믿는다.
        if self.match and not re.search(self.match, str(ev)):
            raise AssertionError(f"message does not match {self.match!r}: {ev}")
        return True


class _Warns:
    """pytest.warns 최소 흉내. 기록을 인덱스로 꺼낼 수 있어야 한다."""

    def __init__(self, category, match=None):
        self.category, self.match = category, match
        self._ctx = None
        self.records: list = []

    def __enter__(self):
        self._ctx = warnings.catch_warnings(record=True)
        self._log = self._ctx.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, et, ev, tb):
        self.records = list(self._log)
        self._ctx.__exit__(et, ev, tb)
        if et is not None:
            return False
        hit = [w for w in self.records if issubclass(w.category, self.category)]
        if not hit:
            raise AssertionError(f"{self.category.__name__} was not raised")
        if self.match and not any(re.search(self.match, str(w.message)) for w in hit):
            raise AssertionError(
                f"no {self.category.__name__} matches {self.match!r}: "
                + "; ".join(str(w.message) for w in hit)
            )
        self.records = hit
        return True

    def __getitem__(self, i):
        return self.records[i]

    def __len__(self):
        return len(self.records)


class _MonkeyPatch:
    """setattr 만 흉내낸다. 되돌리기는 역순으로."""

    def __init__(self):
        self._undo: list = []

    def setattr(self, target, name, value):
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, old in reversed(self._undo):
            setattr(target, name, old)
        self._undo.clear()


class _Mark:
    """@pytest.mark.parametrize 만. 쌓으면 곱집합이 된다 (pytest 와 같다)."""

    @staticmethod
    def parametrize(argnames, argvalues):
        names = [n.strip() for n in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

        def deco(fn):
            sets = getattr(fn, "_shim_params", [])
            fn._shim_params = sets + [(names, list(argvalues))]
            return fn

        return deco

    def __getattr__(self, name):        # 다른 마크는 무시한다
        return lambda *a, **k: (lambda fn: fn)


def _fixture(*args, **kwargs):
    """@pytest.fixture / @pytest.fixture(scope=...) 둘 다 받는다."""

    def mark(fn):
        fn._shim_fixture = kwargs.get("scope", "function")
        return fn

    if args and callable(args[0]):
        return mark(args[0])
    return mark


def _install_shim() -> bool:
    try:
        import pytest  # noqa: F401
        return True
    except ImportError:
        pass
    import types

    m = types.ModuleType("pytest")
    m.approx = lambda expected, rel=1e-6, abs=1e-12: _Approx(expected, rel, abs)
    m.raises = lambda exc, match=None: _Raises(exc, match)
    m.warns = lambda category, match=None: _Warns(category, match)
    m.fixture = _fixture
    m.mark = _Mark()
    m.skip = lambda reason="": (_ for _ in ()).throw(_Skip(reason))
    m.fail = lambda reason="": (_ for _ in ()).throw(AssertionError(reason))
    sys.modules["pytest"] = m
    return False


class _Skip(Exception):
    pass


# ---------------------------------------------------------------------------
def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    # dataclass 는 cls.__module__ 로 sys.modules 를 찾는다. 등록하지 않으면
    # 검사 파일 안에서 @dataclass 를 쓰는 순간 임포트가 통째로 깨진다.
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def _expand(fn):
    """parametrize 를 곱집합으로 편다. 없으면 인자 없는 한 건."""
    sets = getattr(fn, "_shim_params", None)
    if not sets:
        return [{}]
    cases = [{}]
    for names, values in reversed(sets):        # 데코레이터는 아래부터 쌓인다
        out = []
        for base in cases:
            for v in values:
                vals = v if len(names) > 1 else (v,)
                out.append({**base, **dict(zip(names, vals))})
        cases = out
    return cases


def _use_fixture(fn, cache):
    """제너레이터 픽스처면 첫 yield 까지 돌리고 값을 준다."""
    if fn.__name__ in cache:
        got = cache[fn.__name__]
        return got[1] if isinstance(got, tuple) else got
    r = fn()
    if inspect.isgenerator(r):
        value = next(r)
        cache[fn.__name__] = (r, value)
        return value
    cache[fn.__name__] = r
    return r


def _teardown(got):
    if isinstance(got, tuple):
        try:
            next(got[0])
        except StopIteration:
            pass


def run(selectors: list[str]) -> int:
    has_pytest = _install_shim()
    if has_pytest:
        import pytest

        args = [str(TESTS), "-q"]
        return pytest.main(args + list(selectors))

    print("(pytest 미설치 — 내장 러너로 실행합니다)\n")
    files = sorted(TESTS.glob("test_*.py"))
    if selectors:
        files = [f for f in files if any(s in f.stem for s in selectors)]

    passed = failed = skipped = 0
    failures: list[tuple[str, str]] = []

    for f in files:
        try:
            mod = _load(f)
        except Exception:
            failed += 1
            failures.append((f.stem, traceback.format_exc()))
            print(f"{f.stem}: 임포트 실패")
            continue

        fixtures = {n: getattr(mod, n) for n in dir(mod)
                    if hasattr(getattr(mod, n), "_shim_fixture")}
        cache: dict = {}          # 모듈 스코프 픽스처의 값과 뒷정리
        names = [n for n in dir(mod) if n.startswith("test_")]
        line = []
        n_cases = 0
        for name in names:
            fn = getattr(mod, name)
            if not callable(fn) or hasattr(fn, "_shim_fixture"):
                continue
            for case in _expand(fn):
                n_cases += 1
                tmp = None
                mp = _MonkeyPatch()
                try:
                    kwargs = dict(case)
                    params = inspect.signature(fn).parameters
                    if "tmp_path" in params:
                        tmp = Path(tempfile.mkdtemp())
                        kwargs["tmp_path"] = tmp
                    if "monkeypatch" in params:
                        kwargs["monkeypatch"] = mp
                    for pname in params:
                        if pname in fixtures and pname not in kwargs:
                            kwargs[pname] = _use_fixture(fixtures[pname], cache)
                    fn(**kwargs)
                    passed += 1
                    line.append(".")
                except _Skip:
                    skipped += 1
                    line.append("s")
                except Exception:
                    failed += 1
                    line.append("F")
                    tag = f"{f.stem}::{name}"
                    if case:
                        tag += "[" + "-".join(str(v) for v in case.values()) + "]"
                    failures.append((tag, traceback.format_exc()))
                finally:
                    mp.undo()
                    if tmp:
                        shutil.rmtree(tmp, ignore_errors=True)
        for gen in cache.values():          # 모듈 픽스처 뒷정리
            _teardown(gen)
        print(f"{f.stem:<24s} {''.join(line)}  ({n_cases})")

    if failures:
        print("\n" + "=" * 68)
        for name, tb in failures:
            print(f"\nFAILED {name}\n{'-' * 68}\n{tb}")
    print("\n" + "=" * 68)
    print(f"통과 {passed}  실패 {failed}  건너뜀 {skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
