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
def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


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
def _flatten(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, (list, tuple)):
            out[k] = ";".join(str(x) for x in v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out


def save_records(
    records: Sequence[dict], name: str, cfg=None, meta: dict | None = None
) -> Path:
    """레코드 리스트를 outputs/raw/<name>.csv 로 저장하고 메타를 함께 남긴다."""
    ensure_dirs()
    flat = [_flatten(r) for r in records]
    keys: list[str] = []
    for r in flat:
        for k in r:
            if k not in keys:
                keys.append(k)

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
        raise FileNotFoundError(f"{path} 가 없습니다. 해당 실험을 먼저 실행하세요.")
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rows.append({k: _coerce(v) for k, v in raw.items()})
    return rows


def _coerce(v: str) -> Any:
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
        return str(v) if v is not None else "--"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}" if caption else "",
        f"\\label{{{label}}}" if label else "",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(h for _, h in columns) + " \\\\",
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
