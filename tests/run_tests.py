"""pytest 없이도 테스트를 돌리는 러너.

    python tests/run_tests.py            # 전체
    python tests/run_tests.py test_bounds

pytest 가 설치돼 있으면 그쪽으로 위임하고, 없으면 최소 셔임(approx/raises/tmp_path)을
주입해 직접 실행한다. 저장소를 클론하자마자 검증이 가능하게 하려는 장치다.
"""

from __future__ import annotations

import importlib.util
import inspect
import shutil
import sys
import tempfile
import traceback
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

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"{self.exc.__name__} 이 발생하지 않았다")
        if not issubclass(et, self.exc):
            return False
        if self.match and self.match not in str(ev):
            raise AssertionError(f"메시지에 {self.match!r} 가 없다: {ev}")
        return True


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
    spec.loader.exec_module(mod)
    return mod


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

        names = [n for n in dir(mod) if n.startswith("test_")]
        line = []
        for name in names:
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            tmp = None
            try:
                kwargs = {}
                params = inspect.signature(fn).parameters
                if "tmp_path" in params:
                    tmp = Path(tempfile.mkdtemp())
                    kwargs["tmp_path"] = tmp
                fn(**kwargs)
                passed += 1
                line.append(".")
            except _Skip:
                skipped += 1
                line.append("s")
            except Exception:
                failed += 1
                line.append("F")
                failures.append((f"{f.stem}::{name}", traceback.format_exc()))
            finally:
                if tmp:
                    shutil.rmtree(tmp, ignore_errors=True)
        print(f"{f.stem:<24s} {''.join(line)}  ({len(names)})")

    if failures:
        print("\n" + "=" * 68)
        for name, tb in failures:
            print(f"\nFAILED {name}\n{'-' * 68}\n{tb}")
    print("\n" + "=" * 68)
    print(f"통과 {passed}  실패 {failed}  건너뜀 {skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
