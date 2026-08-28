"""실험 결과 저장/로드 + 재현성 스탬핑.

pandas 없이 동작한다 (있으면 편의 함수만 추가로 쓴다).

★ 원칙 ★
실험은 outputs/raw/ 에 원본 레코드를 쓴다.
그림 생성은 **outputs/raw/ 만 읽는다.**
-> 실험을 다시 돌리지 않고도 figure 를 재생성할 수 있다.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.config import PROJECT_ROOT

OUTPUTS = PROJECT_ROOT / "outputs"
RAW_DIR = OUTPUTS / "raw"
FIG_DIR = OUTPUTS / "figures"
TAB_DIR = OUTPUTS / "tables"
LOG_DIR = OUTPUTS / "logs"


def ensure_dirs() -> None:
    for d in (RAW_DIR, FIG_DIR, TAB_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def enable_utf8_stdout() -> None:
    """Windows 콘솔(cp949)에서 한글 출력이 깨지지 않게 한다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 재현성 스탬프
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def git_commit() -> str:
    """짧은 커밋 해시. 작업 트리가 더러우면 -dirty 를 붙인다.

    ★ 커밋되지 않은 변경으로 뽑은 결과가 그 커밋의 것처럼 보이면 안 된다.
      결과 파일 하나하나에 이 값이 박히므로 여기서 구분해 둔다.
    """
    # --match=^$ 는 어떤 태그와도 안 맞는 패턴이다. --always 가 대신 해시를
    # 내주고 --dirty 가 접미사를 붙인다 (태그가 있어도 해시로 통일된다).
    res = _git("describe", "--always", "--dirty", "--match=^$")
    if res:
        return res

    head = _git("rev-parse", "--short", "HEAD")
    if not head:
        return "no-git"
    return f"{head}-dirty" if _git("status", "--porcelain") else head


def provenance(cfg=None, extra: dict | None = None) -> dict:
    """모든 결과 파일에 붙는 메타데이터."""
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
    }
    if cfg is not None:
        meta["config_hash"] = cfg.hash()
        warns = cfg.provenance_warnings()
        meta["n_estimate_params"] = len(warns)
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# 레코드 저장 / 로드
# ---------------------------------------------------------------------------
# 리스트 직렬화 표시. 대괄호가 있어야 문자열과 구분된다.
_L_OPEN, _L_SEP, _L_CLOSE = "[", ";", "]"


def _join_list(v: Iterable[Any]) -> str:
    return _L_OPEN + _L_SEP.join(str(x) for x in v) + _L_CLOSE


def _flatten(rec: dict, parent: str = "", sep: str = ".") -> dict:
    """중첩 dict 를 한 층으로 편다. 리스트는 대괄호로 감싸 직렬화한다."""
    out: dict[str, Any] = {}
    for k, v in rec.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))          # 한 층만 펴면 남는다
        elif isinstance(v, (list, tuple)):
            out[key] = _join_list(v)
        else:
            out[key] = v
    return out


def save_records(
    records: Sequence[dict], name: str, cfg=None, meta: dict | None = None
) -> Path:
    """레코드 리스트를 outputs/raw/<name>.csv 로 저장하고 메타를 함께 남긴다."""
    ensure_dirs()
    flat = [_flatten(r) for r in records]
    keys: list[str] = list(dict.fromkeys(k for r in flat for k in r))

    path = RAW_DIR / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in flat:
            w.writerow(r)

    (RAW_DIR / f"{name}.meta.json").write_text(
        json.dumps(provenance(cfg, meta), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def load_records(name: str) -> list[dict]:
    """outputs/raw/<name>.csv 를 읽어 dict 리스트로. 숫자는 자동 변환."""
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run that experiment first")
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rows.append({k: _coerce(v) for k, v in raw.items()})
    return rows


def _coerce_scalar(v: str) -> Any:
    """CSV 셀 하나를 None / bool / int / float / str 로 되돌린다."""
    if v is None or v == "":
        return None
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _coerce(v: str) -> Any:
    """CSV 셀을 복원한다. 리스트는 _join_list 가 남긴 대괄호로 알아본다."""
    if v is None or v == "":
        return None

    # ★ 맨 세미콜론으로 판별하면 안 된다. "a;b" 같은 평범한 문자열이
    #   조용히 리스트로 바뀌고, 원소가 하나인 리스트는 되돌아오지 못한다.
    if len(v) >= 2 and v[0] == _L_OPEN and v[-1] == _L_CLOSE:
        inner = v[1:-1]
        return [] if inner == "" else [_coerce_scalar(p) for p in inner.split(_L_SEP)]
    return _coerce_scalar(v)


def filter_records(records: Iterable[dict], **conds) -> list[dict]:
    """레코드 필터. filter_records(recs, design='exact', top_k=8)"""
    out = []
    for r in records:
        if all(r.get(k) == v for k, v in conds.items()):
            out.append(r)
    return out


def save_trace(name: str, trace: dict) -> Path:
    """per-step 시계열을 npz 로 저장 (그림용)."""
    import numpy as np

    ensure_dirs()
    path = RAW_DIR / f"{name}.npz"
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in trace.items()})
    return path


# ---------------------------------------------------------------------------
# 논문용 표 (LaTeX / CSV)
# ---------------------------------------------------------------------------
# LaTeX 특수문자. 순차 치환하면 앞서 넣은 \textbackslash{} 의 중괄호를
# 뒤 규칙이 다시 이스케이프해 \textbackslash\{\} 가 된다. 그래서 한 번에 바꾼다.
_LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}
_LATEX_RE = re.compile("|".join(re.escape(c) for c in _LATEX_ESCAPE))


def _escape_latex(text: str) -> str:
    """표에 들어가는 문자열을 LaTeX 안전하게 만든다."""
    return _LATEX_RE.sub(lambda m: _LATEX_ESCAPE[m.group()], text)


def to_latex_table(
    records: Sequence[dict],
    columns: Sequence[tuple[str, str]],
    name: str,
    caption: str = "",
    label: str = "",
    float_fmt: str = "{:.3f}",
) -> Path:
    """columns = [(레코드 키, 표시 헤더), ...]"""
    ensure_dirs()

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return float_fmt.format(v)
        if isinstance(v, (list, tuple)):
            return _escape_latex(", ".join(str(x) for x in v))
        return _escape_latex(str(v)) if v is not None else "--"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_escape_latex(caption)}}}" if caption else "",
        f"\\label{{{label}}}" if label else "",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(_escape_latex(h) for _, h in columns) + " \\\\",
        "\\midrule",
    ]
    for r in records:
        lines.append(" & ".join(fmt(r.get(k)) for k, _ in columns) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    path = TAB_DIR / f"{name}.tex"
    path.write_text("\n".join(x for x in lines if x), encoding="utf-8")

    csv_path = TAB_DIR / f"{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([h for _, h in columns])
        for r in records:
            w.writerow([r.get(k) for k, _ in columns])
    return path
