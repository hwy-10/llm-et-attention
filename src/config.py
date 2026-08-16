"""설정 로더.

PyYAML 이 있으면 그걸 쓰고, 없으면 내장 미니 파서로 동작한다.
config/*.yaml 은 매핑 / 리스트 / 스칼라만 사용하므로 서브셋 파서로 충분하다.
(의도: `pip install` 없이 저장소를 클론하자마자 실험이 돌아가게 한다)
"""

from __future__ import annotations

import hashlib
import json
import re
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


def _mini_yaml_load(text: str) -> dict:
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

    def get(self, dotted: str, default: Any = None) -> Any:
        """'hardware.memory.word_tokens' 처럼 점 표기로 접근."""
        root, _, rest = dotted.partition(".")
        node: Any = getattr(self, root, None)
        if node is None:
            return default
        for part in rest.split("."):
            if not part:
                continue
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

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
                elif k.startswith("source") and v == "estimate":
                    warns.append(f"{path}.{k} = estimate")

        walk(self.hardware, "hardware")
        return warns


def load_config(config_dir: str | Path | None = None) -> Config:
    d = Path(config_dir) if config_dir else CONFIG_DIR
    return Config(**{name: load_yaml(d / f"{name}.yaml") for name in CONFIG_FILES})
