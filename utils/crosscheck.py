"""소프트웨어 예측 ↔ RTL 실측 대조.

소프트웨어 모델(src/schedule.py, src/memory.py)이 낸 사이클·읽기 횟수가
실제 RTL 시뮬레이션과 맞는지 확인한다. 어긋나면 **모델이 틀린 것**이며,
그 상태로 뽑은 절감률은 논문에 쓸 수 없다.

전형적인 불일치 원인
--------------------
  * decision_latency_planes 를 실제 파이프라인 깊이보다 작게 잡음
  * word_tokens (BRAM 워드폭) 가 실제 구현과 다름
  * lanes / batch_size 가 RTL 파라미터와 어긋남
  * 압축 오버헤드(compaction_cost_cycles)를 과소·과대 평가

허용 오차를 넘으면 report() 가 원인 후보를 함께 출력한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hw_parser import parse_sim_csv

MATCH_KEYS = ("design", "seq_len", "top_k", "margin")


@dataclass
class Mismatch:
    key: dict
    metric: str
    predicted: float
    measured: float

    @property
    def rel_error(self) -> float:
        return abs(self.predicted - self.measured) / abs(self.measured) if self.measured else float("inf")

    def __str__(self) -> str:
        k = ", ".join(f"{a}={b}" for a, b in self.key.items())
        return (f"  [{k}] {self.metric}: 예측 {self.predicted:,.0f} vs 실측 "
                f"{self.measured:,.0f}  (오차 {self.rel_error * 100:.1f}%)")


_HINTS = {
    "cycles": [
        "lanes / batch_size 가 RTL 파라미터와 같은지 확인",
        "compaction_cost_cycles 가 실제 압축 오버헤드와 맞는지 확인",
        "파이프라인 채움/비움 구간이 모델에 빠져 있지 않은지 확인",
    ],
    "bram_reads": [
        "word_tokens (BRAM 워드폭) 가 실제 구현과 같은지 확인",
        "decision_latency_planes 를 실제 파이프라인 깊이로 올렸는지 확인",
        "종단 신호가 실제로 읽기 요청을 차단하고 있는지 RTL 파형 확인",
    ],
}


def _key_of(rec: dict) -> tuple:
    return tuple(rec.get(k) for k in MATCH_KEYS)


def compare(
    predicted: list[dict],
    measured: list[dict],
    tolerance: float = 0.05,
    metrics: tuple[str, ...] = ("cycles", "bram_reads"),
) -> dict:
    """예측 레코드와 실측 레코드를 키로 맞춰 비교한다.

    predicted : run_decode() 레코드 (total_cycles, words_bram 사용)
    measured  : rtl_simulation_cycles.csv 파싱 결과 (cycles, bram_reads)
    """
    pred_map = {_key_of(r): r for r in predicted}
    mismatches: list[Mismatch] = []
    matched, unmatched = 0, []

    for m in measured:
        key = _key_of(m)
        p = pred_map.get(key)
        if p is None:
            unmatched.append(dict(zip(MATCH_KEYS, key)))
            continue
        matched += 1
        pairs = {"cycles": ("total_cycles", "cycles"), "bram_reads": ("words_bram", "bram_reads")}
        for metric in metrics:
            pk, mk = pairs[metric]
            if pk not in p or mk not in m or m[mk] is None:
                continue
            mm = Mismatch(dict(zip(MATCH_KEYS, key)), metric, float(p[pk]), float(m[mk]))
            if mm.rel_error > tolerance:
                mismatches.append(mm)

    errs = [mm.rel_error for mm in mismatches]
    return {
        "n_measured": len(measured),
        "n_matched": matched,
        "n_unmatched": len(unmatched),
        "unmatched_keys": unmatched,
        "n_mismatch": len(mismatches),
        "mismatches": mismatches,
        "max_rel_error": float(max(errs)) if errs else 0.0,
        "mean_rel_error": float(np.mean(errs)) if errs else 0.0,
        "tolerance": tolerance,
        "passed": len(mismatches) == 0 and matched > 0,
    }


def report(result: dict, verbose: bool = True) -> str:
    """비교 결과를 사람이 읽는 형태로."""
    lines = [
        "=" * 68,
        "SW 예측 vs RTL 실측 대조",
        "=" * 68,
        f"  실측 레코드   : {result['n_measured']}",
        f"  매칭됨        : {result['n_matched']}",
        f"  매칭 실패     : {result['n_unmatched']}",
        f"  허용 오차     : {result['tolerance'] * 100:.0f}%",
        f"  불일치        : {result['n_mismatch']}",
    ]
    if result["n_mismatch"]:
        lines.append(f"  최대 오차     : {result['max_rel_error'] * 100:.1f}%")
        if verbose:
            lines.append("")
            lines.append("  불일치 항목:")
            for mm in result["mismatches"][:20]:
                lines.append(str(mm))
            hinted = {mm.metric for mm in result["mismatches"]}
            lines.append("")
            lines.append("  점검할 것:")
            for metric in hinted:
                for h in _HINTS.get(metric, []):
                    lines.append(f"    - ({metric}) {h}")
    if result["n_unmatched"]:
        lines.append("")
        lines.append(f"  ⚠ 예측에 없는 실측 조합 {result['n_unmatched']}개 — 스윕 범위를 맞추세요.")

    lines.append("")
    lines.append("  결과: " + ("통과" if result["passed"] else "불일치 — 모델을 고칠 것"))
    lines.append("=" * 68)
    return "\n".join(lines)


def run(predicted: list[dict], sim_csv: str | Path, tolerance: float = 0.05) -> dict:
    """편의 함수: CSV 를 읽어 바로 대조."""
    measured = parse_sim_csv(sim_csv)
    return compare(predicted, measured, tolerance=tolerance)
