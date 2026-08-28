"""정확도·절감률 지표 검증.

이 파일의 수치가 논문의 정확도 주장이 된다. 특히 ``topk_retention`` 은
손실 예산(tests/test_score_budget.py)이 통째로 기대는 함수다.
"""

import warnings

import numpy as np
import pytest

from src.threshold import topk_indices
from utils.metrics import (
    accuracy_metrics,
    attention_distribution_error,
    savings_breakdown,
    softmax,
    spearman_rho,
    termination_profile,
    top1_accuracy,
    topk_retention,
)


# ---------------------------------------------------------------------------
# 상위 k 보존율
# ---------------------------------------------------------------------------

def test_topk_retention_by_hand():
    """겹치는 개수를 손으로 세어 대조."""

    ref = np.array([[9.0, 8.0, 7.0, 6.0, 5.0]])

    # 상위 3 = {0,1,2}. test 도 같으면 만점
    assert topk_retention(ref, ref, 3) == 1.0

    # 순서만 뒤집혀도 집합이 같으면 만점
    assert topk_retention(ref, ref[:, ::-1] * 0 + ref, 3) == 1.0

    # 2등을 꼴찌로 내리면 {0,2,3} 이 되어 2/3
    moved = ref.copy()
    moved[0, 1] = -1.0
    assert topk_retention(ref, moved, 3) == pytest.approx(2 / 3)


def test_terminated_tokens_never_count_as_hits():
    """★ 동률 -inf 가 top-k 에 섞여 보존율을 부풀리지 않는지 검증."""

    rng = np.random.default_rng(1)
    ref = rng.normal(size=(1, 16))

    # 생존 2개, k=8 -> 아무리 잘해도 2/8 이다
    test = ref.copy()
    test[0, np.argsort(ref[0])[:14]] = -np.inf

    got = topk_retention(ref, test, 8)
    assert got <= 2 / 8 + 1e-12, got

    # 전멸한 행은 0점
    dead = np.full_like(ref, -np.inf)
    assert topk_retention(ref, dead, 8) == 0.0
    assert top1_accuracy(ref, dead) == 0.0


def test_shape_mismatch_raises():
    """모양이 다르면 zip 이 짧은 쪽에서 끊겨 없는 행을 만점으로 센다."""

    ref = np.random.default_rng(0).normal(size=(4, 20))
    test = ref.copy()
    test[1:] = -np.inf

    with pytest.raises(ValueError, match="shapes must match"):
        topk_retention(ref[0], test, 4)

    # 제대로 넘기면 3행이 전멸한 것이 점수에 반영된다
    assert topk_retention(ref, test, 4) == pytest.approx(0.25)


def test_metrics_agree_with_the_decode_loop_path():
    """★ src/decode_loop.py 가 같은 계산을 따로 구현하고 있다.

    src 는 utils 를 임포트하지 않는 구조라 중복이 의도된 것인데,
    두 벌이 갈라지면 요약 CSV 와 예산 검사가 다른 말을 하게 된다.
    """
    rng = np.random.default_rng(3)

    for _ in range(200):
        n, k = 32, 8
        ref = rng.normal(size=n)
        test = ref.copy()
        test[rng.choice(n, size=int(rng.integers(0, n - 1)), replace=False)] = -np.inf

        loop = np.intersect1d(topk_indices(ref, k), topk_indices(test, k)).size / k
        assert topk_retention(ref[None, :], test[None, :], k) == pytest.approx(loop)


# ---------------------------------------------------------------------------
# softmax 와 분포 오차
# ---------------------------------------------------------------------------

def test_softmax_handles_a_fully_masked_row_without_warning():
    """전부 종단된 행에서 경고가 새지 않는지 검증."""

    x = np.array([[-np.inf, -np.inf], [1.0, 2.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = softmax(x, axis=1)

    assert np.all(p[0] == 0.0)                 # 분포가 없다
    assert p[1].sum() == pytest.approx(1.0)


def test_kl_separates_the_two_losses():
    """종단 손실과 top-k 희소화 손실을 가르는지 검증."""

    rng = np.random.default_rng(2)
    ref = rng.normal(0, 3.0, size=(4, 32))

    # 오라클 = 참 top-8 만 남긴 희소 어텐션 (종단과 무관한 손실)
    oracle = np.full_like(ref, -np.inf)
    for i in range(4):
        idx = topk_indices(ref[i], 8)
        oracle[i, idx] = ref[i, idx]

    out = attention_distribution_error(ref, oracle, oracle_logits=oracle)

    # test 와 oracle 이 같으면 종단이 더한 몫은 0 이다
    assert out["softmax_kl_terminate"] == pytest.approx(0.0, abs=1e-12)
    assert out["softmax_kl_sparsify"] == pytest.approx(out["softmax_kl"])


def test_kl_absolute_value_depends_on_eps():
    """★ softmax_kl 의 절대값은 eps 가 정한다 — 같은 eps 안에서만 비교할 것."""

    rng = np.random.default_rng(1)
    ref = rng.normal(0, 3.0, size=(4, 32))
    test = ref.copy()
    for i in range(4):
        test[i, np.argsort(ref[i])[:24]] = -np.inf

    seen = [attention_distribution_error(ref, test, eps=e)["softmax_kl"]
            for e in (1e-6, 1e-9, 1e-12)]

    # eps 가 작아질수록 커진다 — 데이터가 아니라 eps 가 만든 값이다
    assert seen[0] < seen[1] < seen[2]

    # 어떤 eps 를 썼는지 결과에 남는다
    assert attention_distribution_error(ref, test)["softmax_kl_eps"] == 1e-12


def test_spearman_ranks_ties_by_index_not_by_average():
    """★ 동률 -inf 를 평균 순위로 다루지 않아 손실 지표로 쓸 수 없다.

    종단된 토큰은 전부 -inf 로 동률인데, argsort 가 그것들에 **인덱스 순서**로
    서로 다른 순위를 준다. 그래서 같은 종단 결과가 배치에 따라 다른 값을 낸다.
    """
    rng = np.random.default_rng(7)
    ref = rng.permutation(32).astype(np.float64)[None, :]

    # 하위 24개를 종단 — 살아남은 8개의 순위는 완전히 보존된다
    test = ref.copy()
    test[0, np.argsort(ref[0])[:24]] = -np.inf

    # 순위를 하나도 안 뒤집었는데 1.0 이 아니다
    assert spearman_rho(ref, test) < 1.0

    # 종단이 하나도 없으면 정상 동작한다 — 문제는 동률뿐이다
    assert spearman_rho(ref, ref) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 절감률 분해
# ---------------------------------------------------------------------------

def test_savings_breakdown_by_hand():
    """분모가 무엇인지 — 이론은 (토큰,평면) 쌍, 실현은 BRAM 워드."""

    summary = {
        "read_saving_ideal": 0.40,      # 살아있는 쌍 기준
        "read_saving_bram": 0.10,       # 워드 기준
        "read_saving_compacted": 0.35,
        "total_cycles": 900,
        "dense_cycles": 1200,
        "cycle_speedup_vs_baseline": 0.13,
    }
    out = savings_breakdown(summary)

    # 실현률 = 실현 / 이론
    assert out["read_realization"] == pytest.approx(0.10 / 0.40)

    # 사이클 절감의 분모는 종단 없는 순차 처리(dense) 다
    assert out["cycle_saving"] == pytest.approx(1.0 - 900 / 1200)


def test_missing_dense_cycles_raises_instead_of_reading_zero():
    """분모가 없으면 절감 0 과 구분이 안 된다."""

    with pytest.raises(KeyError, match="dense_cycles"):
        savings_breakdown({"read_saving_ideal": 0.4, "total_cycles": 900})


def test_termination_profile_by_hand():
    """종단 시점 분포를 손으로 세어 대조."""

    tp = np.array([0, 2, 2, 4, 8, 8, 8, 8])     # 8 = 끝까지 안 죽음
    out = termination_profile(tp, n_planes=8)

    assert out["mean_term_plane"] == pytest.approx(40 / 8)
    assert out["never_terminated_frac"] == pytest.approx(4 / 8)
    assert out["terminated_by_plane2_frac"] == pytest.approx(3 / 8)
    assert out["term_hist_2"] == 2
    assert out["term_hist_8"] == 4


def test_accuracy_metrics_bundles_every_axis():
    """묶음 함수가 어떤 축도 빠뜨리지 않는지."""

    rng = np.random.default_rng(0)
    ref = rng.normal(size=(3, 24))
    out = accuracy_metrics(ref, ref)

    for key in ("top4_retention", "top8_retention", "top16_retention",
                "top1_acc", "spearman", "softmax_kl", "softmax_l1", "ref_top1_mass"):
        assert key in out, key

    # 같은 입력이면 정확도 축은 만점, 분포 오차는 0
    assert out["top8_retention"] == 1.0
    assert out["top1_acc"] == 1.0
    assert out["softmax_kl"] == pytest.approx(0.0, abs=1e-12)


def test_top1_is_not_reported_twice():
    """★ top1_retention 과 top1_acc 는 같은 값이다 — 한 키로만 낸다."""

    rng = np.random.default_rng(4)
    ref = rng.normal(size=(3, 20))
    test = ref.copy()
    test[:, rng.choice(20, size=12, replace=False)] = -np.inf

    # 정의가 같으므로 값도 같다
    assert topk_retention(ref, test, 1) == pytest.approx(top1_accuracy(ref, test))

    # 두 곳 모두 k=1 을 `top1_acc` 한 이름으로만 낸다.
    # decode_loop 쪽은 test_the_decode_summary_uses_the_same_top1_name 이 본다.
    assert "top1_retention" not in accuracy_metrics(ref, test)
    assert "top1_acc" in accuracy_metrics(ref, test)


def test_no_tokens_gives_the_same_answer_on_both_axes():
    """볼 토큰이 없을 때 두 함수가 반대로 답하지 않는지."""

    empty = np.zeros((1, 0))

    assert topk_retention(empty, empty, 8) == top1_accuracy(empty, empty) == 1.0


def test_spearman_checks_shapes_too():
    """모양이 다르면 브로드캐스트로 조용히 통과하던 자리."""

    ref = np.random.default_rng(0).normal(size=(4, 12))

    with pytest.raises(ValueError, match="shapes must match"):
        spearman_rho(ref[0], ref)


def test_termination_profile_rejects_out_of_range_planes():
    """히스토그램이 조용히 잘려 합이 안 맞는 것을 막는다."""

    # n_planes=8 인데 9, 12 가 섞이면 평균에는 들어가고 히스토그램에서는 빠진다
    with pytest.raises(ValueError, match="out of range"):
        termination_profile(np.array([0, 2, 4, 8, 9, 12]), n_planes=8)

    # 정상 범위면 히스토그램 합 == 입력 개수
    tp = np.array([0, 2, 2, 4, 8, 8, 8, 8])
    out = termination_profile(tp, n_planes=8)
    assert sum(v for k, v in out.items() if k.startswith("term_hist_")) == tp.size


def test_savings_breakdown_prefers_the_summary_value():
    """실현률 식이 두 벌이면 언젠가 갈라진다 — 요약에 있으면 그것을 쓴다."""

    s = {"read_saving_ideal": 0.4, "read_saving_bram": 0.1,
         "total_cycles": 900, "dense_cycles": 1200}

    assert savings_breakdown(s)["read_realization"] == pytest.approx(0.25)

    # 요약이 다른 값을 들고 있으면 그것이 이긴다
    s["read_realization_ratio"] = 0.31
    assert savings_breakdown(s)["read_realization"] == pytest.approx(0.31)


def test_the_decode_summary_uses_the_same_top1_name():
    """★ top-1 이 한 이름으로만 나가는지 — 두 경로가 이름을 달리 쓰던 자리.

    `utils/metrics.py` 는 `top1_acc`, `decode_loop` 는 `top1_retention` 을 냈다.
    같은 값인데 예산 검사와 실험 CSV 가 다른 열을 보게 된다.
    """
    from src.dataset import synthetic_qk
    from src.decode_loop import DecodeWorkbench, run_decode

    snap = synthetic_qk(seq_len=96, head_dim=64, seed=0)
    wb = DecodeWorkbench.build(snap, warmup=16, seq_len=96)
    s = run_decode(wb, design="approx", top_k=8, margin=0.9,
                   keep_trace=False, eval_top_k=(1, 8)).summary

    assert "top1_acc" in s
    assert "top1_retention" not in s, "같은 값이 두 이름으로 나간다"

    # 두 경로가 같은 정의인지 — src 는 utils 를 임포트하지 않아 구현이 두 벌이다
    assert 0.0 <= s["top1_acc"] <= 1.0
    assert s["top1_acc"] >= s["top8_retention"], (
        "1등은 상한이 가장 높아 가장 늦게 죽는다 — top1 이 top8 보다 낮으면 전제가 깨진 것이다"
    )
