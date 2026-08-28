"""정확도 · 절감률 지표.  numpy 만 사용한다 (scipy 불필요).

★ 이 파일에서 가장 중요한 함수는 savings_breakdown() 이다 ★
"이론적 절감"과 "BRAM 워드 단위로 실현되는 절감"을 분리해 낸다.
전자만 보고하면 발표에서 바로 지적당한다 (src/memory.py 상단 주석 참조).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.threshold import topk_indices


# ---------------------------------------------------------------------------
# 상위 k 보존율 — 어텐션에서 가장 의미 있는 정확도 지표
# ---------------------------------------------------------------------------
def _pair(ref: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """두 배열을 (행, 토큰) 으로 맞춘다."""

    ref = np.atleast_2d(np.asarray(ref, dtype=np.float64))
    test = np.atleast_2d(np.asarray(test, dtype=np.float64))

    # 모양이 다르면 zip 이 짧은 쪽에서 끊겨 없는 행을 만점으로 세어 버린다
    if ref.shape != test.shape:
        raise ValueError(f"ref {ref.shape} != test {test.shape}: shapes must match")
    return ref, test


def topk_retention(ref: np.ndarray, test: np.ndarray, k: int) -> float:
    """top-k 집합 일치율. 1.0 이면 완전 보존.

    ref/test 는 1-D 또는 2-D(쿼리 x 토큰) 모두 허용.
    종단된 토큰은 test 에서 -inf 로 마스킹되어 있어야 한다.
    """
    ref, test = _pair(ref, test)

    # 볼 토큰이 없으면 잴 것도 없다. top1_accuracy 와 같은 답을 낸다.
    if ref.shape[1] == 0:
        return 1.0
    k = int(min(k, ref.shape[1]))
    if k <= 0:
        return 1.0

    # topk_indices 가 -inf 를 제외한다. 종단된 토큰이 top-k 에 섞이면
    # 보존율이 부풀려지고, 분모는 k 그대로라 모자란 만큼 실점이 된다.
    hits = [
        np.intersect1d(topk_indices(a, k), topk_indices(b, k)).size
        for a, b in zip(ref, test)
    ]
    return float(np.mean(hits) / k)


def top1_accuracy(ref: np.ndarray, test: np.ndarray) -> float:
    """1등이 같은 행의 비율. 전부 종단된 행은 실점이다."""

    ref, test = _pair(ref, test)
    if ref.shape[1] == 0:
        return 1.0

    hit = []
    for a, b in zip(ref, test):
        ta, tb = topk_indices(a, 1), topk_indices(b, 1)
        hit.append(bool(ta.size and tb.size and ta[0] == tb[0]))
    return float(np.mean(hit)) if hit else 1.0


def spearman_rho(ref: np.ndarray, test: np.ndarray) -> float:
    """행별 스피어만 순위상관의 평균 (scipy 없이). -inf 는 최하위로 취급."""
    ref, test = _pair(ref, test)

    # 모양이 다르면 브로드캐스트로 조용히 통과한다 — _pair 가 먼저 막는다
    ref = np.nan_to_num(ref, neginf=-1e300)
    test = np.nan_to_num(test, neginf=-1e300)

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
    hi = np.max(x, axis=axis, keepdims=True)

    # 행이 통째로 -inf 면 (-inf) - (-inf) = nan 이 되면서 경고가 뜬다.
    # 그 행은 어차피 전부 0 으로 나가므로 경고만 막는다.
    with np.errstate(invalid="ignore"):
        z = x - hi
    e = np.exp(np.where(np.isfinite(z), z, -np.inf))
    tot = np.sum(e, axis=axis, keepdims=True)
    return np.divide(e, tot, out=np.zeros_like(e), where=tot > 0)


def attention_distribution_error(
    ref_logits: np.ndarray,
    test_logits: np.ndarray,
    oracle_logits: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict:
    """희소 어텐션(종단 후) vs 조밀 어텐션의 분포 차이.

    ★ softmax_kl 의 절대값은 eps 에 좌우된다 ★
    참조가 질량을 준 토큰이 종단되면 q=0 이라 KL 이 원래 무한대다.
    eps 가 그것을 유한하게 만들 뿐이므로, 같은 eps 안에서의 비교만 뜻이 있다.

    oracle_logits 를 주면 손실을 둘로 가른다.
        oracle = 참 top-k 만 남긴 희소 어텐션 (종단과 무관)
        softmax_kl_sparsify   : top-k 희소화 자체의 손실
        softmax_kl_terminate  : 종단이 더한 몫 = 전체 - 희소화
    """
    ref, test = _pair(ref_logits, test_logits)
    p = softmax(ref, axis=1)
    q = softmax(test, axis=1)

    def _kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sum(a * (np.log(a + eps) - np.log(b + eps)), axis=1)

    kl = _kl(p, q)
    l1 = np.sum(np.abs(p - q), axis=1)
    out = {
        "softmax_kl": float(np.mean(kl)),
        "softmax_l1": float(np.mean(l1)),
        "softmax_l1_max": float(np.max(l1)) if l1.size else 0.0,
        "softmax_kl_eps": float(eps),
        # 참조 분포의 집중도 — 가이드 4.1절 전제가 실제로 성립하는지 확인
        "ref_top1_mass": float(np.mean(np.max(p, axis=1))),
    }

    if oracle_logits is not None:
        _, oracle = _pair(ref_logits, oracle_logits)
        kl_o = _kl(p, softmax(oracle, axis=1))
        out["softmax_kl_sparsify"] = float(np.mean(kl_o))
        out["softmax_kl_terminate"] = float(np.mean(kl - kl_o))
    return out


def accuracy_metrics(
    ref_logits: np.ndarray,
    test_logits: np.ndarray,
    top_k: Sequence[int] = (4, 8, 16),
    oracle_logits: np.ndarray | None = None,
) -> dict:
    """정확도 축을 한 번에 묶는다. 세 축은 각각 다른 방향으로 실패한다."""

    # k=1 은 넣지 않는다 — top1_acc 와 같은 값이 두 키로 나가기 때문이다
    out = {f"top{k}_retention": topk_retention(ref_logits, test_logits, k) for k in top_k}
    out["top1_acc"] = top1_accuracy(ref_logits, test_logits)
    out["spearman"] = spearman_rho(ref_logits, test_logits)
    out.update(attention_distribution_error(ref_logits, test_logits,
                                            oracle_logits=oracle_logits))
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

    # dense_cycles 가 없으면 cycle_saving 이 조용히 0 이 된다 — 절감이 없는 것과
    # 구분이 안 되므로 막는다. run_decode() 요약에는 항상 들어 있다.
    if "dense_cycles" not in summary:
        raise KeyError(
            "summary has no 'dense_cycles'; cycle_saving would silently read 0. "
            f"keys: {sorted(summary)[:12]}"
        )

    return {
        "read_saving_ideal": ideal,
        "read_saving_bram": bram,
        "read_saving_compacted": float(summary.get("read_saving_compacted", 0.0)),
        # 요약이 이미 들고 있으면 그것을 쓴다 — 식이 두 벌이면 언젠가 갈라진다
        "read_realization": float(
            summary.get("read_realization_ratio", (bram / ideal) if ideal > 0 else 1.0)
        ),
        "cycle_saving": 1.0 - float(summary.get("total_cycles", 0))
        / max(float(summary["dense_cycles"]), 1.0),
        "speedup_vs_baseline": float(summary.get("cycle_speedup_vs_baseline", 0.0)),
    }


def termination_profile(term_planes: np.ndarray, n_planes: int = 8) -> dict:
    """종단 시점 분포. 가이드 8.3 1단계의 핵심 관찰값.

    종단이 거의 안 일어나면(mean ≈ n_planes) 설계를 다시 검토해야 한다.
    """
    tp = np.asarray(term_planes, dtype=np.float64).ravel()
    if tp.size == 0:
        return {"mean_term_plane": float(n_planes), "never_terminated_frac": 1.0}

    # 히스토그램은 0..n_planes 로 자른다. 밖의 값이 있으면 합이 안 맞아
    # "몇 개가 어디서 죽었나" 를 못 읽는다.
    if tp.min() < 0 or tp.max() > n_planes:
        raise ValueError(
            f"term_plane out of range [0, {n_planes}]: "
            f"min={tp.min():.0f} max={tp.max():.0f}"
        )
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
