"""정확도 · 절감률 지표.  numpy 만 사용한다 (scipy 불필요).

★ 이 파일에서 가장 중요한 함수는 savings_breakdown() 이다 ★
"이론적 절감"과 "BRAM 워드 단위로 실현되는 절감"을 분리해 낸다.
전자만 보고하면 발표에서 바로 지적당한다 (src/memory.py 상단 주석 참조).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 상위 k 보존율 — 어텐션에서 가장 의미 있는 정확도 지표
# ---------------------------------------------------------------------------
def topk_retention(ref: np.ndarray, test: np.ndarray, k: int) -> float:
    """top-k 집합 일치율. 1.0 이면 완전 보존.

    ref/test 는 1-D 또는 2-D(쿼리 x 토큰) 모두 허용.
    종단된 토큰은 test 에서 -inf 로 마스킹되어 있어야 한다.
    """
    ref = np.atleast_2d(np.asarray(ref, dtype=np.float64))
    test = np.atleast_2d(np.asarray(test, dtype=np.float64))
    k = int(min(k, ref.shape[1]))
    if k <= 0:
        return 1.0
    ri = np.argpartition(-ref, k - 1, axis=1)[:, :k]
    ti = np.argpartition(-test, k - 1, axis=1)[:, :k]
    hits = [np.intersect1d(a, b).size for a, b in zip(ri, ti)]
    return float(np.mean(hits) / k)


def top1_accuracy(ref: np.ndarray, test: np.ndarray) -> float:
    ref = np.atleast_2d(ref)
    test = np.atleast_2d(test)
    return float(np.mean(np.argmax(ref, axis=1) == np.argmax(test, axis=1)))


def spearman_rho(ref: np.ndarray, test: np.ndarray) -> float:
    """행별 스피어만 순위상관의 평균 (scipy 없이). -inf 는 최하위로 취급."""
    ref = np.atleast_2d(np.nan_to_num(ref, neginf=-1e300))
    test = np.atleast_2d(np.nan_to_num(test, neginf=-1e300))

    def ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, axis=1)
        r = np.empty_like(order, dtype=np.float64)
        rows = np.arange(a.shape[0])[:, None]
        r[rows, order] = np.arange(a.shape[1], dtype=np.float64)
        return r

    ra, rb = ranks(ref), ranks(test)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, num / den, 0.0)
    return float(np.mean(rho))


# ---------------------------------------------------------------------------
# softmax 이후 — "softmax 가 오차를 지워준다"는 주장의 검증 (가이드 4.1절)
# ---------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.where(np.isfinite(z), z, -np.inf))
    tot = np.sum(e, axis=axis, keepdims=True)
    return np.divide(e, tot, out=np.zeros_like(e), where=tot > 0)


def attention_distribution_error(ref_logits: np.ndarray, test_logits: np.ndarray) -> dict:
    """희소 어텐션(종단 후) vs 조밀 어텐션의 분포 차이."""
    p = softmax(np.atleast_2d(ref_logits), axis=1)
    q = softmax(np.atleast_2d(test_logits), axis=1)
    eps = 1e-12
    kl = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=1)
    l1 = np.sum(np.abs(p - q), axis=1)
    return {
        "softmax_kl": float(np.mean(kl)),
        "softmax_l1": float(np.mean(l1)),
        "softmax_l1_max": float(np.max(l1)) if l1.size else 0.0,
        # 참조 분포의 집중도 — 가이드 4.1절 전제가 실제로 성립하는지 확인
        "ref_top1_mass": float(np.mean(np.max(p, axis=1))),
    }


def accuracy_metrics(
    ref_logits: np.ndarray,
    test_logits: np.ndarray,
    top_k: Sequence[int] = (1, 4, 8, 16),
) -> dict:
    out = {f"top{k}_retention": topk_retention(ref_logits, test_logits, k) for k in top_k}
    out["top1_acc"] = top1_accuracy(ref_logits, test_logits)
    out["spearman"] = spearman_rho(ref_logits, test_logits)
    out.update(attention_distribution_error(ref_logits, test_logits))
    return out


# ---------------------------------------------------------------------------
# ★ 절감률 분해 — 이론 vs 실현 ★
# ---------------------------------------------------------------------------
def savings_breakdown(summary: dict) -> dict:
    """run_decode() 요약에서 절감 항목만 뽑아 정리한다.

    반환 키
    -------
    read_saving_ideal   : 살아있는 (토큰,평면) 쌍 기준. **이론값**
    read_saving_bram    : 실제 BRAM 워드 읽기 기준. **실현값**
    read_realization    : 실현/이론 비율. 1.0 이면 완전 실현.
    cycle_saving        : 종단 없는 순차 처리(②) 대비 사이클 절감
    speedup_vs_baseline : 기준 설계(①) 대비 사이클 배속 (주파수 미반영)
    """
    ideal = float(summary.get("read_saving_ideal", 0.0))
    bram = float(summary.get("read_saving_bram", 0.0))
    return {
        "read_saving_ideal": ideal,
        "read_saving_bram": bram,
        "read_saving_compacted": float(summary.get("read_saving_compacted", 0.0)),
        "read_realization": (bram / ideal) if ideal > 0 else 1.0,
        "cycle_saving": 1.0 - float(summary.get("total_cycles", 0))
        / max(float(summary.get("dense_cycles", summary.get("total_cycles", 1))), 1.0),
        "speedup_vs_baseline": float(summary.get("cycle_speedup_vs_baseline", 0.0)),
    }


def termination_profile(term_planes: np.ndarray, n_planes: int = 8) -> dict:
    """종단 시점 분포. 가이드 8.3 1단계의 핵심 관찰값.

    종단이 거의 안 일어나면(mean ≈ n_planes) 설계를 다시 검토해야 한다.
    """
    tp = np.asarray(term_planes, dtype=np.float64).ravel()
    if tp.size == 0:
        return {"mean_term_plane": float(n_planes), "never_terminated_frac": 1.0}
    hist = np.bincount(tp.astype(np.int64), minlength=n_planes + 1)[: n_planes + 1]
    out = {
        "mean_term_plane": float(tp.mean()),
        "median_term_plane": float(np.median(tp)),
        "never_terminated_frac": float(np.mean(tp >= n_planes)),
        "terminated_by_plane2_frac": float(np.mean(tp <= 2)),
        "terminated_by_plane4_frac": float(np.mean(tp <= 4)),
    }
    for m in range(n_planes + 1):
        out[f"term_hist_{m}"] = int(hist[m])
    return out


# ---------------------------------------------------------------------------
# 언어모델 지표 (실제 캡처 경로에서만)
# ---------------------------------------------------------------------------
def perplexity_from_logprobs(logprobs: np.ndarray) -> float:
    return float(np.exp(-np.mean(np.asarray(logprobs, dtype=np.float64))))


def perplexity_delta(ppl_ref: float, ppl_test: float) -> dict:
    return {
        "ppl_ref": ppl_ref,
        "ppl_test": ppl_test,
        "ppl_delta": ppl_test - ppl_ref,
        "ppl_delta_pct": 100.0 * (ppl_test - ppl_ref) / ppl_ref if ppl_ref else float("nan"),
    }
