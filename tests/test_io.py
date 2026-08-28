"""실험 결과 저장·로드와 재현성 스탬프 검증.

여기가 틀리면 CSV 는 멀쩡해 보이는데 값의 **타입**이 조용히 바뀐다.
"""

import subprocess

import pytest

from utils.io import (
    _coerce,
    _escape_latex,
    _flatten,
    _join_list,
    git_commit,
    load_records,
    save_records,
    to_latex_table,
)


# ---------------------------------------------------------------------------
# LaTeX 이스케이프
# ---------------------------------------------------------------------------

def test_escape_does_not_re_escape_its_own_output():
    r"""★ 순차 치환이면 백슬래시 치환문의 중괄호를 뒤 규칙이 또 바꾼다.

    `\` -> `\textbackslash{}` 로 바꾼 뒤 `{`/`}` 규칙이 다시 걸리면
    `\textbackslash\{\}` 가 되어 LaTeX 에서 글자 그대로 찍힌다.
    """
    assert _escape_latex("a\\b") == r"a\textbackslash{}b"
    assert r"\{\}" not in _escape_latex("a\\b")

    # 물결·캐럿도 중괄호를 달고 나온다 — 같은 함정이다
    assert _escape_latex("a~b") == r"a\textasciitilde{}b"
    assert _escape_latex("x^2") == r"x\textasciicircum{}2"


def test_escape_covers_the_characters_that_actually_appear():
    """설계 이름과 정책 이름에 밑줄이 들어간다 — 안 막으면 LaTeX 이 깨진다."""

    assert _escape_latex("seq_no_et") == r"seq\_no\_et"
    assert _escape_latex("two_phase") == r"two\_phase"
    for raw, want in (("a&b", r"a\&b"), ("50%", r"50\%"), ("100$", r"100\$"),
                      ("#1", r"\#1"), ("{k}", r"\{k\}")):
        assert _escape_latex(raw) == want, raw


def test_latex_table_escapes_headers_and_cells(tmp_path, monkeypatch):
    """머리글·캡션·셀 전부 통과해야 한다. 하나만 빠져도 컴파일이 죽는다."""

    import utils.io as io_mod
    monkeypatch.setattr(io_mod, "TAB_DIR", tmp_path)

    path = to_latex_table(
        [{"policy": "two_phase", "ratio": 0.5}],
        [("policy", "정책_이름"), ("ratio", "비율")],
        "t", caption="50% 지점",
    )
    text = path.read_text(encoding="utf-8")

    assert r"two\_phase" in text
    assert r"정책\_이름" in text
    assert r"50\%" in text


# ---------------------------------------------------------------------------
# 값의 타입이 왕복하는가
# ---------------------------------------------------------------------------

def test_list_round_trips_including_the_awkward_sizes():
    """원소 1개와 빈 리스트가 특히 잘 깨진다."""

    for orig in ([1, 2, 3], [5], [], ["a", "b"], [1.5, 2.5]):
        assert _coerce(_join_list(orig)) == orig, orig


def test_a_plain_string_with_a_semicolon_stays_a_string():
    """★ 세미콜론만 보고 쪼개면 평범한 문자열이 조용히 리스트가 된다.

    대괄호로 감싼 것만 리스트로 되돌린다. 감싸지 않은 세미콜론은 데이터다.
    """
    for s in ("a;b", "batch_size=8;16", "two_phase", "0.5;1.0"):
        assert _coerce(s) == s, s


def test_scalars_keep_their_type():
    assert _coerce("") is None
    assert _coerce("true") is True and _coerce("False") is False
    assert _coerce("12") == 12 and isinstance(_coerce("12"), int)
    assert _coerce("1.5") == 1.5
    assert _coerce("exact") == "exact"


def test_flatten_goes_all_the_way_down():
    """한 층만 펴면 두 층째 dict 가 str(dict) 로 CSV 에 박힌다."""

    got = _flatten({"a": 1, "b": {"c": {"d": 2}}, "e": [1, 2]})
    assert got == {"a": 1, "b.c.d": 2, "e": "[1;2]"}


def test_save_then_load_keeps_types(tmp_path, monkeypatch):
    """CSV 를 한 바퀴 돌려도 값이 그대로인지."""

    import utils.io as io_mod
    monkeypatch.setattr(io_mod, "RAW_DIR", tmp_path)

    recs = [{"design": "exact", "cycles": 32969, "ratio": 0.9573,
             "win": False, "ks": [4, 8, 16], "nested": {"a": {"b": 1}}}]
    save_records(recs, "t")
    back = load_records("t")[0]

    assert back["design"] == "exact"
    assert back["cycles"] == 32969 and isinstance(back["cycles"], int)
    assert back["ratio"] == pytest.approx(0.9573)
    assert back["win"] is False
    assert back["ks"] == [4, 8, 16]
    assert back["nested.a.b"] == 1


# ---------------------------------------------------------------------------
# 재현성 스탬프
# ---------------------------------------------------------------------------

def test_git_commit_marks_a_dirty_tree(tmp_path, monkeypatch):
    """★ 커밋 안 된 변경으로 뽑은 결과가 그 커밋의 것처럼 보이면 안 된다."""

    import utils.io as io_mod

    repo = tmp_path / "r"
    repo.mkdir()

    def run(*a):
        subprocess.run(list(a), cwd=repo, capture_output=True, text=True, timeout=10)

    run("git", "init", "-q")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")
    (repo / "f.txt").write_text("1", encoding="utf-8")
    run("git", "add", ".")
    run("git", "commit", "-q", "-m", "x")

    monkeypatch.setattr(io_mod, "PROJECT_ROOT", repo)
    clean = git_commit()
    if clean == "no-git":
        pytest.skip("git not available")
    assert not clean.endswith("-dirty"), clean

    (repo / "f.txt").write_text("2", encoding="utf-8")
    dirty = git_commit()
    assert dirty.endswith("-dirty"), dirty
    assert dirty.startswith(clean), (clean, dirty)


def test_git_commit_never_raises_outside_a_repo(tmp_path, monkeypatch):
    """실험 도중 죽으면 안 된다 — 못 알아내면 문자열로 그렇다고 적는다."""

    import utils.io as io_mod
    monkeypatch.setattr(io_mod, "PROJECT_ROOT", tmp_path)
    assert isinstance(git_commit(), str)
