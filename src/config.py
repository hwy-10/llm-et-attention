"""설정 로더.

PyYAML 이 있으면 그걸 쓰고, 없으면 내장 미니 파서로 동작한다.
config/*.yaml 은 매핑 / 리스트 / 스칼라만 사용하므로 서브셋 파서로 충분하다.
(의도: `pip install` 없이 저장소를 클론하자마자 실험이 돌아가게 한다)
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILES = ("model", "quant", "hardware", "sweeps")


# ---------------------------------------------------------------------------
# 미니 YAML 파서 (서브셋)
# ---------------------------------------------------------------------------
_INLINE_LIST = re.compile(r"^\[(.*)\]$")


def _split_top_level(s: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _parse_scalar(tok: str) -> Any:
    tok = tok.strip()
    if not tok:
        return None
    if tok[0] in "\"'" and tok[-1] == tok[0] and len(tok) >= 2:
        return tok[1:-1]
    low = tok.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    m = _INLINE_LIST.match(tok)
    if m:
        inner = m.group(1).strip()
        return [] if not inner else [_parse_scalar(p) for p in _split_top_level(inner)]
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


class ConfigKeyError(KeyError):
    """설정 경로가 없는데 조용히 기본값으로 넘어가려 할 때."""


def _mini_yaml_load(text: str) -> dict:
    # ★ 탭은 PyYAML 이 ScannerError 로 막는데 이 파서는 조용히 통과시켜
    #   들여쓰기가 무시된 **다른 구조**를 만든다. 에러가 없어서 더 위험하다.
    #   PyYAML 유무에 따라 결과가 갈리면 안 되므로 여기서도 막는다.
    for i, raw in enumerate(text.splitlines(), 1):
        stripped = raw.split("#", 1)[0]
        if stripped[: len(stripped) - len(stripped.lstrip())].count("	"):
            raise ValueError(
                f"{i}행: 들여쓰기에 탭이 있다. YAML 은 공백만 허용한다 "
                f"(PyYAML 은 에러를 내지만 이 파서는 조용히 다른 구조를 만든다)"
            )
    lines = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))
    if not lines:
        return {}

    def build(idx: int, indent: int):
        if idx < len(lines) and lines[idx][1].startswith("- "):
            items = []
            while idx < len(lines) and lines[idx][0] == indent and lines[idx][1].startswith("- "):
                items.append(_parse_scalar(lines[idx][1][2:]))
                idx += 1
            return items, idx

        mapping: dict[str, Any] = {}
        while idx < len(lines):
            cur_indent, content = lines[idx]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                idx += 1
                continue
            if ":" not in content:
                idx += 1
                continue
            key, _, rest = content.partition(":")
            key, rest = key.strip(), rest.strip()
            if rest:
                mapping[key] = _parse_scalar(rest)
                idx += 1
            else:
                nxt = idx + 1
                if nxt < len(lines) and lines[nxt][0] > cur_indent:
                    child, idx = build(nxt, lines[nxt][0])
                    mapping[key] = child
                else:
                    mapping[key] = {}
                    idx += 1
        return mapping, idx

    result, _ = build(0, lines[0][0])
    return result if isinstance(result, dict) else {"_root": result}


def load_yaml(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        return _mini_yaml_load(text)


# ---------------------------------------------------------------------------
@dataclass
class Config:
    """config/ 아래 네 파일을 하나로 묶은 설정."""

    model: dict = field(default_factory=dict)
    quant: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)
    sweeps: dict = field(default_factory=dict)

    # --- 자주 쓰는 값 바로가기 -------------------------------------------
    @property
    def head_dim(self) -> int:
        return int(self.model["model"]["head_dim"])

    @property
    def n_planes(self) -> int:
        return int(self.quant["planes"]["n_planes"])

    @property
    def seq_len(self) -> int:
        return int(self.model["decode"]["seq_len"])

    @property
    def warmup_tokens(self) -> int:
        return int(self.model["decode"]["warmup_tokens"])

    def get(self, dotted: str, default: Any = None, *, required: bool = False) -> Any:
        """'hardware.memory.word_tokens' 처럼 점 표기로 접근.

        ★ 기본 동작은 '없으면 조용히 default' 다. 이게 실제 사고를 냈다 —
          `quant.quant.n_planes`(존재하지 않는 경로)가 기본값 8 로 통과했는데
          진짜 값도 8 이라 아무도 몰랐다.

        `required=True` 를 주면 경로가 없을 때 ConfigKeyError 를 낸다.
        **오타가 조용히 묻히면 안 되는 자리에는 이걸 쓸 것.**
        """
        root, _, rest = dotted.partition(".")
        node: Any = getattr(self, root, None)
        missing = node is None
        if not missing:
            for part in rest.split("."):
                if not part:
                    continue
                if not isinstance(node, dict) or part not in node:
                    missing = True
                    break
                node = node[part]
        if missing:
            if required:
                raise ConfigKeyError(f"설정 경로 {dotted!r} 가 없다")
            return default
        return node

    def require(self, dotted: str) -> Any:
        """`get(dotted, required=True)` 의 짧은 이름."""
        return self.get(dotted, required=True)

    # --- 재현성 --------------------------------------------------------
    def hash(self) -> str:
        blob = json.dumps(
            {k: getattr(self, k) for k in CONFIG_FILES}, sort_keys=True, default=str
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def provenance_warnings(self) -> list[str]:
        """source 가 estimate 인 하드웨어 파라미터를 모아 경고한다.

        발표에서 "그 숫자 어디서 나왔냐" 질문에 대비하는 장치.
        RTL 실측이 들어오면 이 목록이 비어야 한다.
        """
        warns: list[str] = []

        def walk(node: Any, path: str) -> None:
            if not isinstance(node, dict):
                return
            for k, v in node.items():
                if isinstance(v, dict):
                    walk(v, f"{path}.{k}" if path else k)
                elif k.startswith("source") and isinstance(v, str) and v.strip().lower().startswith("estimat"):
                    # ★ 예전에는 v == "estimate" 정확 일치만 봤다. "Estimate" ·
                    #   "estimated" · "estimate " 같은 오타가 전부 통과해
                    #   **실측으로 교체된 것처럼 보였다.**
                    warns.append(f"{path}.{k} = {v}")

        walk(self.hardware, "hardware")
        return warns


def load_config(config_dir: str | Path | None = None) -> Config:
    d = Path(config_dir) if config_dir else CONFIG_DIR
    return Config(**{name: load_yaml(d / f"{name}.yaml") for name in CONFIG_FILES})


# 설정 -> dataclass 배선
# yaml 값이 dataclass 기본값과 전부 같아 파싱이 실패해도 숫자가 안 변한다.
# 기본값으로 흘러내릴 때마다 경고를 남기는 이유다.


class ConfigDefaultWarning(UserWarning):
    """설정 항목이 없어 기본값을 대신 썼다."""


def as_bool(raw: Any) -> bool:
    """yaml 의 참/거짓."""

    # bool("false") == True 사고를 막는다
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low in ("true", "yes", "on", "1"):
            return True
        if low in ("false", "no", "off", "0"):
            return False
    if isinstance(raw, int):
        return bool(raw)
    raise ValueError(f"{raw!r} is not a boolean (use true / false)")


# 배선표 한 줄 = (dataclass 필드, yaml 점표기, 변환 함수)
Wiring = tuple[tuple[str, str, Any], ...]


def read_fields(cfg: Config, cls: type, wiring: Wiring) -> dict[str, Any]:
    """배선표대로 cfg 에서 값을 뽑아 cls(**...) 에 넣을 dict 를 만든다."""

    # 기본값은 dataclass 에서 가져온다 — 두 군데 적으면 언젠가 어긋난다
    defaults = cls()
    out: dict[str, Any] = {}
    seen: dict[str, dict] = {}

    for fname, dotted, cast in wiring:
        sec_path, _, key = dotted.rpartition(".")

        if sec_path not in seen:
            node = cfg.get(sec_path)
            if not isinstance(node, dict) or not node:
                missing = sorted(
                    d.rpartition(".")[2] for _f, d, _c in wiring
                    if d.rpartition(".")[0] == sec_path
                )
                warnings.warn(
                    ConfigDefaultWarning(
                        f"{cls.__name__}: config section '{sec_path}' is missing or "
                        f"empty; falling back to defaults for {missing}. "
                        f"Check config/{sec_path.split('.')[0]}.yaml."
                    ),
                    stacklevel=3,
                )
                node = {}
            seen[sec_path] = node

        node = seen[sec_path]
        default = getattr(defaults, fname)

        if key in node:
            raw = node[key]
            try:
                out[fname] = cast(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{dotted} = {raw!r} cannot be used for "
                    f"{cls.__name__}.{fname}: {exc}"
                ) from exc
            continue

        if node:  # 섹션은 있는데 이 키만 없다 -> 오타일 가능성이 높다
            near = difflib.get_close_matches(key, list(node), n=3, cutoff=0.6)
            hint = (f"did you mean '{near[0]}'?" if near
                    else f"keys present in this section: {sorted(node)}")
            warnings.warn(
                ConfigDefaultWarning(
                    f"{cls.__name__}.{fname}: '{dotted}' is missing; "
                    f"falling back to default {default!r}. {hint}"
                ),
                stacklevel=3,
            )
        out[fname] = default

    return out


def apply_overrides(obj: Any, overrides: dict[str, Any]) -> Any:
    """replace(obj, **overrides) — 오타면 쓸 수 있는 항목을 알려 준다."""

    if not overrides:
        return obj
    valid = {f.name for f in dataclasses.fields(obj)}
    bad = sorted(set(overrides) - valid)
    if bad:
        raise TypeError(
            f"{type(obj).__name__} has no field {bad}; valid fields: {sorted(valid)}"
        )
    return dataclasses.replace(obj, **overrides)
