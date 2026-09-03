"""조기 종단 판정 루프 검증 — src/terminator.py

★ 이 파일이 생기기 전까지 terminator.py 를 임포트하는 테스트가 하나도 없었다.
  src/designs.py 와 src/schedule.py 만 이 모듈을 불렀고, 검증은 run_design /
  run_decode 를 거치는 통합 경로에만 의존했다.

기준 예제는 docs/background/attention_walkthrough.md 의 4토큰 케이스다.
그 문서의 모든 숫자를 손으로 검산해 두었으므로, 여기서도 **하드코딩된 기대값**을
그대로 쓴다. 값이 바뀌면 문서와 코드 중 하나가 틀린 것이다.
"""

import numpy as np
import pytest

from src.bounds import step_bounds
from src.masked_sum import partial_dots
from src.quantize import N_PLANES, to_bitplanes
from src.terminator import (
    PlaneDecision,
    TerminationController,
    masked_scores,
    run_step,
    run_step_from_frontend,
)
from src.threshold import ThetaPolicy

# --- attention_walkthrough.md 3-1 --------------------------------------------
_Q = np.array([5, -3, 7, 2], dtype=np.int64)
_K = np.array(
    [[200, 32, 176, 16],
     [96, 224, 48, 128],
     [232, 16, 208, 64],
     [24, 160, 40, 224]], dtype=np.int64
)
# 참 점수 (문서 3-2)
_TRUE = _K @ _Q                                   # [2168, 400, 2696, 368]


def _case():
    p = partial_dots(_Q, to_bitplanes(_K, N_PLANES))[:, 0, :]   # (8, 4)
    return p, step_bounds(_Q, N_PLANES)


def _run(top_k=2, margin=0.0, latency=0, policy="every_plane", enable=True, **kw):
    p, b = _case()
    return run_step(
        p, b,
        ThetaPolicy(name=policy, top_k=top_k, margin=margin, margin_mode="relative_width", **kw),
        decision_latency=latency, enable_termination=enable,
    )


# ---------------------------------------------------------------------------
# 0. 기준 예제가 문서와 같은가
# ---------------------------------------------------------------------------
def test_setup_matches_the_walkthrough_document():
    p, b = _case()
    assert (b.q_pos, b.q_neg) == (14, -3)
    np.testing.assert_array_equal(p[0], [12, -1, 12, -1])       # b7 평면
    np.testing.assert_array_equal(p[1], [5, 2, 14, 2])          # b6 평면
    np.testing.assert_array_equal(_TRUE, [2168, 400, 2696, 368])


# ---------------------------------------------------------------------------
# 1. 평면 순서와 누산
# ---------------------------------------------------------------------------
def test_plane_loop_is_msb_first():
    """평면 0 이 b7 이어야 한다. 순서가 뒤집히면 상한식이 서지 않는다."""
    p, _ = _case()
    # 128 자리 비트가 1인 토큰은 0, 2 번 -> b7 부분합이 크다
    assert p[0][0] == p[0][2] == 12
    assert p[0][1] == p[0][3] == -1


def test_survivor_scores_are_exact():
    """끝까지 살아남은 토큰의 점수는 참값과 비트 단위로 같아야 한다."""
    r = _run()
    np.testing.assert_array_equal(r.s_int[r.alive], _TRUE[r.alive])


def test_no_termination_reproduces_the_full_dot_product():
    """설계 ② (enable_termination=False) 는 전 토큰의 참값을 낸다."""
    r = _run(enable=False)
    np.testing.assert_array_equal(r.s_int, _TRUE)
    assert r.alive.all()
    np.testing.assert_array_equal(r.term_plane, np.full(4, N_PLANES))
    assert r.read_live.all(), "종단이 없으면 모든 평면을 읽어야 한다"


# ---------------------------------------------------------------------------
# 2. ★ 종단 시점 — 문서 4-4 와 정확히 같아야 한다
# ---------------------------------------------------------------------------
def test_terminates_exactly_where_the_document_says():
    """문서 4-4: θ=1667 에서 t1, t3 의 상한 882 가 못 미쳐 평면 2에서 종단."""
    r = _run(top_k=2)
    np.testing.assert_array_equal(r.alive, [True, False, True, False])
    np.testing.assert_array_equal(r.term_plane, [N_PLANES, 2, N_PLANES, 2])
    assert r.theta_trace[1] == 1667.0, f"평면 2 시점 θ 가 {r.theta_trace[1]} (문서 1667)"
    assert r.theta_trace[0] == 1155.0, f"평면 1 시점 θ 가 {r.theta_trace[0]} (문서 1155)"


def test_true_topk_survives():
    """★ 무손실 판정 ★ 참 top-2 = {t2, t0} 가 전부 살아남아야 한다."""
    r = _run(top_k=2)
    true_top = set(np.argsort(-_TRUE)[:2].tolist())
    assert true_top <= set(np.flatnonzero(r.alive).tolist())


def test_frozen_scores_stop_accumulating():
    """종단된 토큰의 누산은 그 시점에 동결된다.

    문서 4-4 기준 t1, t3 은 S_2 = 0 에서 멈춘다. 참값(400, 368)까지 가지 않는다.
    """
    r = _run(top_k=2)
    assert r.s_int[1] == 0 and r.s_int[3] == 0
    assert r.s_int[1] != _TRUE[1], "동결이 안 되면 참값이 나온다"


# ---------------------------------------------------------------------------
# 3. ★ 판정 지연 — 읽기 절감의 과대평가를 막는 장치
# ---------------------------------------------------------------------------
def test_read_live_counts_match_the_document_when_latency_is_zero():
    """문서 4-5: 평면 읽기 32장 -> 20장 (t1, t3 은 2장씩만).

    지연 0 이면 t=0,1 은 4토큰씩, t=2..7 은 2토큰씩 -> 4+4+12 = 20.
    """
    r = _run(top_k=2, latency=0)
    assert int(r.read_live.sum()) == 20
    assert r.read_live.size == 32
    np.testing.assert_array_equal(r.read_live.sum(axis=1), [4, 4, 2, 2, 2, 2, 2, 2])


def test_latency_shifts_the_mask_by_exactly_one_plane():
    """지연 1 이면 평면 t 는 live_history[t-1] 을 쓴다 -> 읽기가 2장 늘어 22장."""
    r = _run(top_k=2, latency=1)
    assert int(r.read_live.sum()) == 22
    np.testing.assert_array_equal(r.read_live.sum(axis=1), [4, 4, 4, 2, 2, 2, 2, 2])


def test_more_latency_never_increases_savings():
    counts = [int(_run(top_k=2, latency=l).read_live.sum()) for l in range(N_PLANES + 1)]
    assert counts == sorted(counts), f"지연이 커질수록 읽기가 늘어야 한다: {counts}"
    assert counts[0] == 20


def test_latency_at_or_beyond_n_planes_erases_all_savings():
    """★ 조용한 실패 지점 — 지연이 평면 수 이상이면 마스크가 항상 초기값이다.

    read_live 가 전부 True 가 되어 절감이 0 이 되는데 경고가 없다.
    """
    r = _run(top_k=2, latency=N_PLANES)
    assert r.read_live.all()
    assert int(r.read_live.sum()) == 32
    # 그래도 점수와 종단 판정 자체는 바뀌지 않는다 (지연은 읽기 회계에만 걸린다)
    ref = _run(top_k=2, latency=0)
    np.testing.assert_array_equal(r.s_int, ref.s_int)
    np.testing.assert_array_equal(r.alive, ref.alive)


def test_latency_does_not_change_the_answer():
    """★ 하드웨어 설정은 답을 바꾸지 않는다 — 읽을 양만 바꾼다."""
    ref = _run(top_k=2, latency=0)
    for lat in (1, 2, 5):
        r = _run(top_k=2, latency=lat)
        np.testing.assert_array_equal(r.s_int, ref.s_int)
        np.testing.assert_array_equal(r.alive, ref.alive)
        np.testing.assert_array_equal(r.term_plane, ref.term_plane)


# ---------------------------------------------------------------------------
# 4. keep-top-k 가드 — 한 번 잘못 구현됐던 자리
# ---------------------------------------------------------------------------
def test_never_drops_below_topk_even_with_absurd_margin():
    """수치 동률이나 과격한 margin 으로 전부 잘리는 사고가 없어야 한다."""
    for margin in (0.0, 1.0, 5.0, 100.0):
        for k in (1, 2, 3, 4):
            r = _run(top_k=k, margin=margin)
            assert int(r.alive.sum()) >= min(k, 4), f"margin={margin} k={k}"


def test_guard_does_not_disable_pruning_wholesale():
    """★ 예전 결함 — 가드를 '전부-아니면-전무'로 걸면 큰 margin 에서 가지치기가
    통째로 무효화된다. margin 을 키웠을 때 종단이 늘어나는지로 확인한다.
    """
    prev = None
    for margin in (0.0, 0.5, 2.0):
        r = _run(top_k=2, margin=margin)
        cur = float(r.term_plane.mean())
        if prev is not None:
            assert cur <= prev + 1e-9, f"margin={margin} 에서 종단이 오히려 늦어졌다"
        prev = cur


# ---------------------------------------------------------------------------
# 5. 출력 형태
# ---------------------------------------------------------------------------
def test_masked_scores_excludes_terminated_tokens():
    r = _run(top_k=2)
    ms = masked_scores(r)
    assert np.isneginf(ms[~r.alive]).all()
    np.testing.assert_array_equal(ms[r.alive], _TRUE[r.alive])


def test_step_result_shapes_and_ranges():
    r = _run(top_k=2)
    assert r.read_live.shape == (N_PLANES, 4)
    assert r.theta_trace.shape == (N_PLANES,)
    assert r.live_count.shape == (N_PLANES,)
    assert r.s_int.shape == r.alive.shape == r.term_plane.shape == (4,)
    assert np.all((r.term_plane >= 0) & (r.term_plane <= N_PLANES))
    assert 0.0 <= r.survivor_frac <= 1.0


def test_bad_partials_shape_is_rejected():
    _, b = _case()
    try:
        run_step(np.zeros(8), b, ThetaPolicy(top_k=2))
    except ValueError:
        return
    raise AssertionError("1차원 partials 가 조용히 통과했다")


# ---------------------------------------------------------------------------
# 6. ★ 경계 — upper 가 θ 와 정확히 같을 때는 죽이면 안 된다
# ---------------------------------------------------------------------------
def test_upper_equal_to_theta_must_survive():
    """★ 돌연변이 시험에서 유일하게 안 잡히던 구멍 ★

    `upper < theta` 를 `upper <= theta` 로 바꿔도 저장소의 어떤 테스트도 실패하지
    않았다. 무작위 데이터에서 동률이 나오지 않기 때문이다.

    그런데 `<=` 는 **무손실이 아니다.** upper == theta 인 토큰은 실제 점수가 정확히
    theta 일 수 있고, 그러면 상위 k 에 공동으로 들 자격이 있다. 죽이면 참 top-k 를
    잃는다.

    동률을 일부러 만든다. q_pos=64 이면 R_2 = 63x64 = 4032 가 64의 배수라
    S_2(=128,64 가중합)와 정확히 같은 값을 만들 수 있다.

        토큰0 : P_b7=31, P_b6=1  ->  S_2 = 128x31 + 64x1 = 4032  = θ (k=1)
        토큰1 : P_b7=0,  P_b6=0  ->  S_2 = 0,  U_2 = 0 + 4032    = 4032  ← 동률
    """
    from src.bounds import StepBounds

    p = np.zeros((N_PLANES, 2), dtype=np.int64)
    p[0] = [31, 0]
    p[1] = [1, 0]
    b = StepBounds(q_pos=64, q_neg=0, n_planes=N_PLANES)

    # 전제 확인 — 정말 동률인가
    s_2 = 128 * 31 + 64 * 1
    assert s_2 == b.r(2) == 4032, "동률 구성이 깨졌다"

    r = run_step(p, b, ThetaPolicy(name="every_plane", top_k=1, margin=0.0))
    assert r.term_plane[1] != 2, (
        "upper == theta 인 토큰이 평면 2에서 죽었다 -> 비교가 <= 로 바뀌었다"
    )
    assert r.term_plane[1] == 3, f"평면 3에서 종단되어야 한다 (실제 {r.term_plane[1]})"


# ---------------------------------------------------------------------------
# 7. ★ prev_step 이 손실을 내는 메커니즘 — 문헌에 분석이 없는 자리
# ---------------------------------------------------------------------------
def test_guard_alone_prevents_the_naive_prev_step_loss():
    """직전 θ 가 높다는 것만으로는 손실이 안 난다 — 가드가 막는다.

    이 성질을 모르면 prev_step 의 손실 조건을 잘못 서술하게 된다.
    """
    q = np.array([1, 0], dtype=np.int64)
    K = np.array([[255, 0], [0, 255]], dtype=np.int64)
    p = partial_dots(q, to_bitplanes(K, N_PLANES))[:, 0, :]
    b = step_bounds(q, N_PLANES)
    true_top = int(np.argmax(K @ q))

    r = run_step(p, b, ThetaPolicy(name="prev_step", top_k=1, margin=0.0),
                 prev_theta=25500.0)          # 이번 스텝 최대 점수(255)보다 훨씬 높다
    assert r.alive[true_top], "가드가 참 top-1 을 되살려야 한다"


def test_prev_step_loses_when_partial_order_differs_from_true_order():
    """★ prev_step 의 진짜 손실 조건 — 두 가지가 겹쳐야 한다.

      ① 직전 θ 가 이번 스텝 점수보다 높다   -> 전부 종단 대상이 된다
      ② 그 시점의 부분합 순서가 참 순서와 다르다

    가드는 '상한이 큰 순'으로 되살리는데, U_m = S_m + W_m·Q+ 이고 W_m·Q+ 는 모든
    토큰에 같으므로 **상한 순서 = 부분합 순서**다. 초반 평면의 부분합 순서가 참
    순서와 어긋나면 가드가 엉뚱한 토큰을 살린다.

        q = [1, -1]
        A = [128, 127]   b7=1 이라 S_1 이 크지만 하위비트가 깎아내린다 -> 참값 1
        B = [127,   0]   b7=0 이라 S_1 은 0 이지만                    -> 참값 127
    """
    q = np.array([1, -1], dtype=np.int64)
    K = np.array([[128, 127], [127, 0]], dtype=np.int64)
    s = K @ q
    p = partial_dots(q, to_bitplanes(K, N_PLANES))[:, 0, :]
    b = step_bounds(q, N_PLANES)

    # 전제 — 참 순서와 부분합 순서가 반대여야 반례가 성립한다
    assert s[1] > s[0], "참 top-1 은 B 여야 한다"
    assert 128 * p[0][0] > 128 * p[0][1], "평면1 부분합은 A 가 커야 한다"

    true_top = int(np.argmax(s))                      # = 1 (B)
    sound = run_step(p, b, ThetaPolicy(name="every_plane", top_k=1, margin=0.0))
    assert sound.alive[true_top], "every_plane 은 무손실이어야 한다"

    lossy = run_step(p, b, ThetaPolicy(name="prev_step", top_k=1, margin=0.0),
                     prev_theta=300.0)                # U_1 이 둘 다 못 넘는 값
    assert not lossy.alive[true_top], (
        "prev_step 이 참 top-1 을 잃어야 한다 — 잃지 않으면 손실 조건이 바뀐 것이다"
    )


# ---------------------------------------------------------------------------
# 8. ★ 가드의 성질 — 되살리기만 한다 / 건전한 θ 에서는 안 뜬다
# ---------------------------------------------------------------------------
def test_guard_never_fires_under_a_sound_theta():
    """★ 건전한 θ + 정확 모드에서 가드는 죽은 코드다.

    θ 가 '활성 토큰 하한 중 k번째'이면 lower >= θ 인 토큰이 최소 k개 있고,
    그 토큰들은 upper >= lower >= θ 라 살아남는다. 따라서 survivors >= k 이고
    가드 조건(survivors < top_k)이 성립할 수 없다.

    실측: 실제 Llama 텐서 3,360회 판정에서 every_plane / once_at_m / oracle_fixed
    전부 0회. prev_step 만 13.6% 뜬다.

    ⚠ RTL 함의 — prev_step 이나 근사 모드를 안 쓸 거면 회로에서 빼도 된다.
    """
    import src.terminator as term

    fired = []
    orig = term.topk_indices
    term.topk_indices = lambda s, k: (fired.append(1), orig(s, k))[1]
    try:
        for seed in range(6):
            rng = np.random.default_rng(seed)
            K = rng.integers(0, 256, (40, 8))
            q = rng.integers(-127, 128, 8)
            p = partial_dots(q, to_bitplanes(K, N_PLANES))[:, 0, :]
            b = step_bounds(q, N_PLANES)
            for pol in ("every_plane", "once_at_m"):
                run_step(p, b, ThetaPolicy(name=pol, top_k=8, margin=0.0, once_at_m=4))
    finally:
        term.topk_indices = orig
    assert not fired, f"건전한 θ 에서 가드가 {len(fired)}회 발동했다 — 성질이 바뀐 것이다"


def test_margin_one_keeps_exactly_topk_by_lower_bound():
    """★ margin=1.0 (relative_width) 은 '하한 상위 k개만 남기기'와 같다.

        kill ⟺ upper < θ + 1.0·width(m)
        upper = lower + width(m)  이므로  ⟺  lower < θ
        θ 가 하한 중 k번째 → 하한 상위 정확히 k개가 남는다

    이것이 절감이 포화하는 이유다. margin > 1.0 은 더 줄이지 못한다.
    """
    rng = np.random.default_rng(0)
    K = rng.integers(0, 256, (60, 8))
    q = rng.integers(-127, 128, 8)
    p = partial_dots(q, to_bitplanes(K, N_PLANES))[:, 0, :]
    b = step_bounds(q, N_PLANES)
    k = 8

    r = run_step(p, b, ThetaPolicy(name="every_plane", top_k=k, margin=1.0,
                                   margin_mode="relative_width"))
    # 평면 1 판정 직후 생존 = 평면 2 를 읽는 토큰 (지연 0 기준)
    alive_after_p1 = set(np.flatnonzero(r.read_live[2]).tolist())
    lower_at_1 = 128 * p[0] + b.l_offset(1)
    assert alive_after_p1 == set(np.argsort(-lower_at_1)[:k].tolist())
    assert len(alive_after_p1) == k

    # margin 을 더 키워도 생존 수가 줄지 않는다 (바닥)
    r2 = run_step(p, b, ThetaPolicy(name="every_plane", top_k=k, margin=3.0,
                                    margin_mode="relative_width"))
    assert int(r2.alive.sum()) >= k


# ===========================================================================
# 스트리밍 인터페이스 (TerminationController) — 2026-08-29 채택분
#
# run_step 은 8평면을 통째로 돌고 결과만 준다. RTL 파형과 대조하려면
# **평면마다** 중간값을 봐야 하므로 스트리밍 경로를 함께 둔다.
# 두 경로가 갈라질 수 있으므로 여기서 같음을 못 박는다.
# ===========================================================================

def _rand_step(rng, T=60, k=16, margin=0.0, planes=N_PLANES):
    K = rng.integers(0, 256, (T, 8))
    q = rng.integers(-127, 128, 8)
    p = partial_dots(q, to_bitplanes(K, planes))[:, 0, :]
    b = step_bounds(q, planes)
    pol = ThetaPolicy(top_k=k, margin=margin, margin_mode="relative_width")
    return p, b, pol


def test_streaming_equals_batch():
    """★ 평면마다 돌린 결과가 run_step 과 같아야 한다. 한쪽만 고치면 갈라진다."""
    rng = np.random.default_rng(11)
    for _ in range(40):
        T = int(rng.integers(1, 120))
        lat = int(rng.integers(0, 4))
        p, b, pol = _rand_step(rng, T=T, k=int(rng.choice([4, 16, 32])),
                               margin=float(rng.choice([0.0, 0.5])))
        batch = run_step(p, b, pol, decision_latency=lat)

        ctrl = TerminationController(b, pol, T, decision_latency=lat)
        while not ctrl.done:
            ctrl.process_plane(p[ctrl.next_plane])
        stream = ctrl.finish()

        for f in ("s_int", "alive", "term_plane", "read_live", "live_count"):
            np.testing.assert_array_equal(
                getattr(batch, f), getattr(stream, f),
                err_msg="%s (T=%d, lat=%d)" % (f, T, lat),
            )


def test_plane_decision_reports_what_actually_happened():
    """PlaneDecision 이 그 평면의 상태를 정확히 담는가.

    이 값들로 RTL 파형을 대조할 것이므로 하나라도 어긋나면 안 된다.
    """
    rng = np.random.default_rng(5)
    p, b, pol = _rand_step(rng, T=60, k=16)
    ctrl = TerminationController(b, pol, 60, decision_latency=1)
    decs = []
    while not ctrl.done:
        decs.append(ctrl.process_plane(p[ctrl.next_plane]))
    res = ctrl.finish()

    assert all(isinstance(d, PlaneDecision) for d in decs)
    assert [d.plane_index for d in decs] == list(range(N_PLANES))
    assert all(d.m == d.plane_index + 1 for d in decs)

    for d in decs:
        # 구간이 뒤집히면 상한식이 깨진 것이다
        assert np.all(d.lower <= d.upper), d.plane_index
        # 죽은 토큰은 죽기 전에 살아 있어야 한다
        assert np.all(d.live_before[d.killed])
        np.testing.assert_array_equal(d.live_after, d.live_before & ~d.killed)
        # read_mask 는 StepResult 의 그 평면 행과 같아야 한다
        np.testing.assert_array_equal(d.read_mask, res.read_live[d.plane_index])

    # 평면을 볼수록 구간이 좁아진다 (W_m 이 절반씩 준다)
    widths = [float(np.max(d.upper - d.lower)) for d in decs]
    assert all(widths[i + 1] <= widths[i] for i in range(len(widths) - 1)), widths
    # 마지막 평면에서는 점으로 수렴 — 상한 = 하한 = 참값
    np.testing.assert_array_equal(decs[-1].lower, decs[-1].upper)


def test_controller_rejects_misuse():
    """상태를 들고 있는 객체이므로 오용을 막아야 한다."""
    rng = np.random.default_rng(1)
    p, b, pol = _rand_step(rng, T=10)

    with pytest.raises(ValueError, match="n_tokens"):
        TerminationController(b, pol, -1)
    with pytest.raises(ValueError, match="latency"):
        TerminationController(b, pol, 10, decision_latency=-1)

    c = TerminationController(b, pol, 10)
    with pytest.raises(ValueError, match="shape"):
        c.process_plane(np.zeros(5))          # 토큰 수가 다르다
    with pytest.raises(RuntimeError, match="not finished"):
        c.finish()                            # 평면이 남았다
    for t in range(N_PLANES):
        c.process_plane(p[t])
    with pytest.raises(RuntimeError, match="already"):
        c.process_plane(p[0])                 # 9번째 평면


# ===========================================================================
# 팀1 -> 팀2 인계 함수
# ===========================================================================

def test_frontend_computes_bounds_once_and_matches_manual_path():
    """run_step_from_frontend 가 q_stored 로 Q+/Q- 를 맞게 만드는가."""
    rng = np.random.default_rng(3)
    for _ in range(30):
        T = int(rng.integers(1, 80))
        K = rng.integers(0, 256, (T, 8))
        q = rng.integers(-127, 128, 8)
        p = partial_dots(q, to_bitplanes(K, N_PLANES))[:, 0, :]

        auto = run_step_from_frontend(q, p, top_k=16, margin=0.0,
                                      margin_mode="relative_width",
                                      decision_latency=1)
        manual = run_step(p, step_bounds(q, N_PLANES),
                          ThetaPolicy(top_k=16, margin=0.0,
                                      margin_mode="relative_width"),
                          decision_latency=1)
        np.testing.assert_array_equal(auto.s_int, manual.s_int)
        np.testing.assert_array_equal(auto.alive, manual.alive)
        assert auto.extra["q_pos"] == int(q[q > 0].sum())
        assert auto.extra["q_neg"] == int(q[q < 0].sum())


def test_frontend_defaults_follow_theta_policy_not_its_own():
    """★ 편의 함수가 자체 기본값을 들고 있으면 설정이 조용히 무시된다.

    초판은 top_k=8 / once_at_m=3 / margin_mode="relative_gap" 을 들고 있었는데
    셋 다 우리가 이미 고친 값의 **옛 버전**이었다. 특히 relative_gap 은
    relative_width 로 확정하고 ThetaPolicy 기본값까지 바꾼 것을 되돌리고 있었다.
    """
    # ★ 두 모드가 실제로 갈리는 조건을 골라 둔다 (seed 0 / T 50 / k 8 / margin 0.5).
    #   아래 마지막 단언이 그 갈림을 확인하므로, 조건이 무뎌지면 바로 걸린다.
    rng = np.random.default_rng(0)
    K = rng.integers(0, 256, (50, 8))
    q = rng.integers(-127, 128, 8)
    p = partial_dots(q, to_bitplanes(K, N_PLANES))[:, 0, :]

    # ★ margin 을 0 이 아닌 값으로 준다 ★
    #   margin=0 이면 relative_gap 과 relative_width 가 똑같이 0 을 내므로
    #   기본값이 뭐든 결과가 같아 시험이 아무것도 안 막는다.
    d = ThetaPolicy()
    assert d.margin_mode == "relative_width", "확정값이 바뀌면 이 시험도 고칠 것"

    auto = run_step_from_frontend(q, p, top_k=8, margin=0.5, decision_latency=1)
    explicit = run_step(
        p, step_bounds(q, N_PLANES),
        ThetaPolicy(name=d.name, top_k=8, once_at_m=d.once_at_m,
                    margin=0.5, margin_mode=d.margin_mode),
        decision_latency=1,
    )
    np.testing.assert_array_equal(auto.alive, explicit.alive)
    np.testing.assert_array_equal(auto.s_int, explicit.s_int)

    # 옛 기본값(relative_gap)이었다면 다른 답이 나온다 — 그래야 이 시험이 값어치가 있다
    wrong = run_step(
        p, step_bounds(q, N_PLANES),
        ThetaPolicy(name=d.name, top_k=8, once_at_m=d.once_at_m,
                    margin=0.5, margin_mode="relative_gap"),
        decision_latency=1,
    )
    assert not np.array_equal(auto.alive, wrong.alive), (
        "relative_gap 과 relative_width 가 같은 답을 내면 이 시험이 무의미하다 "
        "— margin 이나 데이터를 바꿀 것"
    )


def test_plane_count_mismatch_is_rejected():
    """★ bounds 와 partials 의 평면 수가 다르면 조용히 틀린 답이 나온다.

    partials 가 **더 적을 때** 예전 구현은 예외 없이 통과했다.
    l_offset(m) 이 다른 가중치를 써서 상한식이 성립하지 않는다.
    """
    q = np.ones(8, dtype=np.int16)
    p8 = np.zeros((8, 10), dtype=np.int64)
    p4 = np.zeros((4, 10), dtype=np.int64)
    pol = ThetaPolicy(top_k=4)

    with pytest.raises(ValueError, match="plane-count mismatch"):
        run_step(p4, step_bounds(q, 8), pol)
    with pytest.raises(ValueError, match="plane-count mismatch"):
        run_step(p8, step_bounds(q, 4), pol)


# ===========================================================================
# ★ 종단된 토큰의 확정 점수 — 돌연변이 시험이 찾아낸 구멍 (2026-08-29)
#
#   frozen[killed] = 0 으로 바꿔도 416개 테스트가 **하나도** 안 잡았다.
#   기존 테스트는 전부 생존 토큰만 보거나 두 실행을 비교했다.
#   이 값은 designs.DesignResult.scores_raw 로 새어 나간다.
# ===========================================================================

def test_terminated_token_keeps_its_partial_score_at_termination():
    """종단된 토큰의 s_int 는 **종단 시점의 부분 점수 S_m** 이어야 한다.

    0 도 아니고 최종값도 아니다. scores_raw 로 노출되므로 못 박는다.
    """
    from src.accumulator import cumulative_accumulate

    rng = np.random.default_rng(4)
    p, b, pol = _rand_step(rng, T=80, k=16)
    r = run_step(p, b, pol, decision_latency=1)
    dead = ~r.alive
    assert dead.any(), "종단이 안 일어나면 이 시험이 무의미하다"

    cum = cumulative_accumulate(p)      # (n_planes+1, T), index m = m개 처리 후
    for j in np.flatnonzero(dead):
        m = int(r.term_plane[j])
        assert 0 < m < N_PLANES, "t%d term_plane=%d" % (j, m)
        assert r.s_int[j] == cum[m][j], (
            "t%d: s_int=%d 인데 종단 시점 S_%d=%d"
            % (j, r.s_int[j], m, cum[m][j])
        )
    # 살아남은 토큰은 끝까지 누산한 값이다
    for j in np.flatnonzero(r.alive):
        assert r.s_int[j] == cum[N_PLANES][j]


def test_terminated_score_is_neither_zero_nor_final():
    """0 으로 두거나 최종값으로 두면 둘 다 틀리다 — 양쪽을 막는다."""
    from src.accumulator import cumulative_accumulate

    rng = np.random.default_rng(6)
    p, b, pol = _rand_step(rng, T=100, k=8)
    r = run_step(p, b, pol, decision_latency=0)
    dead = np.flatnonzero(~r.alive)
    assert dead.size >= 3

    final = cumulative_accumulate(p)[N_PLANES]
    # 부분 점수가 최종값과 다른 토큰이 실제로 있어야 이 시험이 성립한다
    differs = [j for j in dead if r.s_int[j] != final[j]]
    assert differs, "전부 최종값과 같으면 아무것도 안 막는다"
    assert not np.all(r.s_int[dead] == 0)
