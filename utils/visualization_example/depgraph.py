"""SW 검증 담당 범위의 코드 상관관계 — import 그래프.

교보재다. 코어 시뮬레이터는 이 모듈을 임포트하지 않는다.

손으로 그린 그림은 코드가 바뀌면 조용히 틀린다. 그래서 저장소를 ``ast`` 로
직접 파싱해 **실제 import 관계**만 그린다. 노드 위치까지 여기서 계산하고
브라우저는 그리기만 한다 (이 패키지의 규약).

담는 것
------
* 담당 구역별 색 — 핵심 / 추가 할당 / 통합 몫 / 상대 팀
* 모듈 간 import 간선 (무엇을 가져다 쓰는지 이름까지)
* 테스트 -> 모듈 덮개 간선
* 줄 수와 한 줄 역할
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

__all__ = ["build", "scan", "SCOPES"]

PKG_DIRS = ("src", "utils", "experiments", "tests")

#: 담당 구역. Notion 분담표 그대로다 — 여기가 바뀌면 그림도 바뀐다.
SCOPES: dict[str, dict[str, str]] = {
    "core":        {"label": "핵심 · 비트평면 + 부분내적", "short": "핵심"},
    "extra":       {"label": "추가 할당 · 회계·비용",       "short": "추가"},
    "integration": {"label": "통합 몫 · 우리 팀",           "short": "통합"},
    "bridge":      {"label": "브리지 — 상대 팀이 만든다",   "short": "브리지"},
    "other":       {"label": "상대 팀 · 범위 밖",           "short": "범위 밖"},
    "test":        {"label": "검증 코드",                   "short": "검증"},
}

_OWN: dict[str, str] = {
    # 핵심 5개
    "src/quantize.py": "core",
    "src/masked_sum.py": "core",
    "src/accumulator.py": "core",
    "src/memory.py": "core",
    "src/schedule.py": "core",
    # 7절 추가 할당
    "utils/metrics.py": "extra",
    "utils/cost_model.py": "extra",
    "experiments/exp4_schedule_policy.py": "extra",
    "experiments/exp6_breakeven.py": "extra",
    # 통합 페이지 B — 우리 팀 몫
    "src/designs.py": "integration",
    "utils/hw_parser.py": "integration",
    "utils/crosscheck.py": "integration",
    "utils/visualization.py": "integration",
    # 브리지 — read_live 를 만드는 쪽
    "src/terminator.py": "bridge",
}

#: 한 줄 역할. 없는 파일은 빈 문자열로 둔다 (그림에는 이름만 나온다).
_ROLE: dict[str, str] = {
    "src/quantize.py": "K 를 unsigned INT8 로, 8개 비트평면으로",
    "src/masked_sum.py": "곱셈 없는 부분 내적 P_b",
    "src/accumulator.py": "시프트 누산 S_m · zero-point 보정",
    "src/memory.py": "★ BRAM 워드 단위 읽기 회계",
    "src/schedule.py": "읽기 차단 · 작업 압축 · 사이클",
    "src/designs.py": "네 설계를 한 인터페이스로",
    "utils/metrics.py": "정확도 · 절감률 지표",
    "utils/cost_model.py": "자원·주파수 감안한 손익분기",
    "utils/hw_parser.py": "Vivado 보고서 · RTL 로그 파싱",
    "utils/crosscheck.py": "SW 예측 ↔ RTL 실측 대조",
    "utils/visualization.py": "논문용 그림 생성",
    "experiments/exp4_schedule_policy.py": "스케줄 정책 × 워드폭 함정",
    "experiments/exp6_breakeven.py": "손익분기 스윕",
    "src/terminator.py": "종단 판정 → read_live 생성",
    "src/threshold.py": "θ 정책",
    "src/bounds.py": "잔여 상·하한",
    "src/config.py": "yaml 로더 · 배선표",
    "src/decode_loop.py": "T 가 자라는 디코드 루프",
    "utils/io.py": "결과 저장 · 재현성 스탬프",
}

#: 이 배열 하나가 두 팀의 경계다. import 되는 이름이 아니라 StepResult 의
#: **속성**이므로 import 목록에는 안 나온다 — 원문에서 찾아야 한다.
BRIDGE_NAME = "read_live"


# ---------------------------------------------------------------------------
# 저장소 파싱
# ---------------------------------------------------------------------------
def _module_map(root: Path) -> dict[str, str]:
    """import 이름 -> 저장소 상대 경로."""
    out: dict[str, str] = {}
    for d in PKG_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for f in sorted(base.glob("*.py")):
            rel = f"{d}/{f.name}"
            name = rel[:-3].replace("/", ".")
            out[name] = rel
            if name.endswith(".__init__"):     # from experiments import X
                out[name[: -len(".__init__")]] = rel
    return out


def _targets(node: ast.AST, pkg: str) -> list[str]:
    """import 문 하나가 가리키는 모듈 이름들."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level:                          # 상대 import
            return [f"{pkg}.{node.module}" if node.module else pkg]
        return [node.module] if node.module else []
    return []


def scan(root: Path | str | None = None) -> dict[str, Any]:
    """저장소를 파싱해 (파일 -> 정보) 와 간선 목록을 만든다."""
    root = Path(root) if root else PROJECT_ROOT
    mods = _module_map(root)
    if not mods:
        raise FileNotFoundError(f"no python packages under {root}; expected {PKG_DIRS}")

    files = sorted({v for v in mods.values()})
    info: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for rel in files:
        path = root / rel
        text = path.read_text(encoding="utf-8", errors="replace")
        info[rel] = {
            "id": rel,
            "label": Path(rel).stem,
            "dir": rel.split("/")[0],
            "lines": text.count("\n") + 1,
            "role": _ROLE.get(rel, ""),
            "scope": "test" if rel.startswith("tests/") else _OWN.get(rel, "other"),
            "touches_bridge": BRIDGE_NAME in text,
        }
        try:
            tree = ast.parse(text)
        except SyntaxError:                     # 파싱 실패는 그래프에서 빠질 뿐이다
            continue

        pkg = rel.split("/")[0]
        merged: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            for name in _targets(node, pkg):
                dst = mods.get(name)
                if not dst or dst == rel:
                    continue
                got = [a.name for a in node.names] if isinstance(node, ast.ImportFrom) else []
                merged.setdefault(dst, []).extend(got)
        for dst, names in merged.items():
            edges.append({
                "source": rel,
                "target": dst,
                "names": sorted(dict.fromkeys(names)),
                "kind": "test" if rel.startswith("tests/") else "import",
            })
    return {"info": info, "edges": edges}


# ---------------------------------------------------------------------------
# 배치 — 의존 깊이를 열로 쓴다
# ---------------------------------------------------------------------------
def _levels(nodes: list[str], edges: list[dict[str, Any]]) -> dict[str, int]:
    """0 = 아무것도 안 부르는 잎. 부르는 것이 있으면 그중 가장 깊은 것 + 1."""
    dep: dict[str, set[str]] = {n: set() for n in nodes}
    for e in edges:
        if e["source"] in dep and e["target"] in dep:
            dep[e["source"]].add(e["target"])

    lvl: dict[str, int] = {}

    def walk(u: str, seen: frozenset[str]) -> int:
        if u in lvl:
            return lvl[u]
        if u in seen:                            # 순환은 0 으로 끊는다
            return 0
        deeper = [walk(v, seen | {u}) for v in sorted(dep[u])]
        lvl[u] = 1 + max(deeper) if deeper else 0
        return lvl[u]

    for n in nodes:
        walk(n, frozenset())
    return lvl


NODE_W, NODE_H = 162, 50
COL_GAP, ROW_GAP = 214, 74
PAD_X, PAD_Y = 24, 88


def _place(nodes: list[str], lvl: dict[str, int], info: dict[str, Any]) -> dict[str, Any]:
    """열 = 의존 깊이, 행 = 같은 깊이 안의 순서."""
    cols: dict[int, list[str]] = {}
    for n in nodes:
        cols.setdefault(lvl[n], []).append(n)

    order = {"core": 0, "extra": 1, "integration": 2, "bridge": 3, "other": 4, "test": 5}
    pos: dict[str, dict[str, int]] = {}
    for c, members in cols.items():
        members.sort(key=lambda n: (order[info[n]["scope"]], info[n]["label"]))
        for r, n in enumerate(members):
            pos[n] = {
                "x": PAD_X + c * COL_GAP,
                "y": PAD_Y + r * ROW_GAP,
                "col": c,
                "row": r,
            }
    width = PAD_X * 2 + (max(lvl.values()) + 1) * COL_GAP - (COL_GAP - NODE_W)
    height = PAD_Y + max(len(m) for m in cols.values()) * ROW_GAP + 24
    return {"pos": pos, "width": width, "height": height, "n_cols": max(lvl.values()) + 1}


# ---------------------------------------------------------------------------
def build(root: Path | str | None = None, include_tests: bool = True) -> dict[str, Any]:
    """화면이 그릴 그래프 전체."""
    raw = scan(root)
    info, edges = raw["info"], raw["edges"]

    # 우리 범위 + 그 범위가 직접 닿는 파일만 남긴다. 전체를 다 그리면 안 읽힌다.
    ours = {p for p, v in info.items() if v["scope"] in ("core", "extra", "integration")}
    keep = set(ours)
    for e in edges:
        if e["kind"] == "test":
            continue
        if e["source"] in ours:
            keep.add(e["target"])
        if e["target"] in ours:
            keep.add(e["source"])

    tests: dict[str, list[str]] = {p: [] for p in keep}
    if include_tests:
        for e in edges:
            if e["kind"] == "test" and e["target"] in ours:
                tests[e["target"]].append(e["source"])

    nodes = sorted(keep)
    kept_edges = [
        e for e in edges
        if e["kind"] == "import" and e["source"] in keep and e["target"] in keep
    ]
    lvl = _levels(nodes, kept_edges)
    layout = _place(nodes, lvl, info)

    deg_in = {n: 0 for n in nodes}
    deg_out = {n: 0 for n in nodes}
    for e in kept_edges:
        deg_out[e["source"]] += 1
        deg_in[e["target"]] += 1

    out_nodes = []
    for n in nodes:
        v = dict(info[n])
        v.update(layout["pos"][n])
        v["level"] = lvl[n]
        v["fan_in"] = deg_in[n]
        v["fan_out"] = deg_out[n]
        v["tests"] = sorted(Path(t).stem for t in tests.get(n, []))
        out_nodes.append(v)

    out_edges = []
    for e in kept_edges:
        crosses = info[e["source"]]["scope"] != info[e["target"]]["scope"]
        # BRIDGE_NAME 을 e["names"] 에서 찾던 조건이 있었는데 한 번도 안 걸렸다.
        # 브리지는 "우리 코드가 판정 결과를 받아 가는 지점" 이다. terminator 가
        # 자기 의존을 부르는 간선은 상대 팀 내부라 여기 들지 않는다.
        out_edges.append({
            **e,
            "bridge": info[e["target"]]["scope"] == "bridge",
            "crosses_scope": crosses,
        })

    return {
        "nodes": out_nodes,
        "edges": out_edges,
        "scopes": SCOPES,
        "layout": {k: layout[k] for k in ("width", "height", "n_cols")},
        "node_size": {"w": NODE_W, "h": NODE_H},
        "summary": _summary(out_nodes, out_edges),
    }


def _summary(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    by_scope: dict[str, dict[str, int]] = {}
    for n in nodes:
        s = by_scope.setdefault(n["scope"], {"files": 0, "lines": 0})
        s["files"] += 1
        s["lines"] += n["lines"]
    ours = [n for n in nodes if n["scope"] in ("core", "extra", "integration")]
    return {
        "by_scope": by_scope,
        "our_files": len(ours),
        "our_lines": sum(n["lines"] for n in ours),
        "edges": len(edges),
        "bridge_edges": sum(1 for e in edges if e["bridge"]),
        "bridge_name": BRIDGE_NAME,
        "touch_bridge": sorted(n["label"] for n in nodes if n["touches_bridge"]),
        "untested": sorted(n["label"] for n in ours if not n["tests"]),
    }
