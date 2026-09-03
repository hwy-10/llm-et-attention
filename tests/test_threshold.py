"""θ 관리부 검증 — src/threshold.py

★ 이 파일이 생기기 전까지 threshold.py 를 직접 겨냥한 테스트가 하나도 없었다.
  kth_largest / margin_abs / ThetaTracker / topk_indices 는 run_design 을 거치는
  간접 경로로만 덮여 있었고, 그래서 우리 모듈의 실패가 상대 팀 테스트에서 터졌다.

검증 대상은 네 가지다.
  1. θ 가 "활성 토큰 하한 중 k번째"라는 정의를 지키는가
  2. margin 정규화 4종이 각각 의도한 스케일을 쓰는가
  3. 정책 4종의 갱신 시점이 맞는가  (★ once_at_m 경계 포함)
  4. topk_indices 가 -inf 를 어떻게 다루는가
"""

import numpy as np

from src.quantize import N_PLANES, remaining_scale
from src.threshold import (
    MARGIN_MODES,
    POLICIES,
    ThetaPolicy,
    ThetaTracker,
    kth_largest,
    topk_indices,
)


# ---------------------------------------------------------------------------
# 1. kth_largest — θ 의 정의 그 자체
# ---------------------------------------------------------------------------
def test_kth_largest_hand_calc():
    v = np.array([10.0, 50.0, 20.0, 40.0, 30.0])
    live = np.ones(5, dtype=bool)
    assert kth_largest(v, live, 1) == 50.0
    assert kth_largest(v, live, 2) == 40.0
    assert kth_largest(v, live, 5) == 10.0


def test_kth_largest_counts_only_live_tokens():
    """★ 죽은 토큰은 θ 계산에 들어가면 안 된다.

    들어가면 θ 가 실제보다 높아지고, 살아 있어야 할 토큰이 잘린다.
    """
    v = np.array([10.0, 50.0, 20.0, 40.0, 30.0])
    live = np.array([True, False, True, False, True])   # 살아있는 값: 10, 20, 30
    assert kth_largest(v, live, 1) == 30.0
    assert kth_largest(v, live, 2) == 20.0
    assert kth_largest(v, live, 3) == 10.0


def test_kth_largest_returns_neg_inf_when_not_enough_live():
    """활성 토큰이 k 개 미만이면 θ 를 정할 수 없다 -> -inf (= 종단 없음)."""
    v = np.array([10.0, 50.0, 20.0])
    live = np.array([True, False, False])
    assert kth_largest(v, live, 2) == -np.inf
    assert kth_largest(v, live, 3) == -np.inf
    assert kth_largest(v, np.zeros(3, dtype=bool), 1) == -np.inf
    # k <= 0 도 마찬가지
    assert kth_largest(v, np.ones(3, dtype=bool), 0) == -np.inf


def test_kth_largest_handles_ties():
    v = np.array([7.0, 7.0, 7.0, 1.0])
    live = np.ones(4, dtype=bool)
    assert kth_largest(v, live, 3) == 7.0
    assert kth_largest(v, live, 4) == 1.0


# ---------------------------------------------------------------------------
# 2. margin 정규화
# ---------------------------------------------------------------------------
def test_default_margin_mode_is_relative_width():
    """★ 2026-08-28 확정값. 실험 전부가 이 모드로 돈다.

    예전에는 ThetaPolicy 기본값만 relative_gap 이라 문서와 코드가 어긋나 있었다.
    run_design / run_decode 의 기본값과 여기가 반드시 같아야 한다.
    """
    from src.decode_loop import run_decode
    from src.designs import run_design
    import inspect

    assert ThetaPolicy().margin_mode == "relative_width"
    assert inspect.signature(run_design).parameters["margin_mode"].default == "relative_width"
    assert inspect.signature(run_decode).parameters["margin_mode"].default == "relative_width"


def test_zero_margin_is_zero_in_every_mode():
    """★ 정확 모드에서는 4종이 전부 같아야 한다.

    이것이 'margin=0 인 approx 는 exact 와 완전히 같다'는 불변식의 근거다.
    """
    for mode in MARGIN_MODES:
        p = ThetaPolicy(margin=0.0, margin_mode=mode)
        assert p.margin_abs(width=12345.0, q_pos=678.0, gap=99.0) == 0.0
        assert p.is_exact


def test_margin_modes_use_their_own_scale():
    width, q_pos, gap = 1000.0, 50.0, 7.0
    assert ThetaPolicy(margin=0.5, margin_mode="relative_gap").margin_abs(width, q_pos, gap) == 3.5
    assert ThetaPolicy(margin=0.5, margin_mode="relative_width").margin_abs(width, q_pos, gap) == 500.0
    assert ThetaPolicy(margin=0.5, margin_mode="absolute").margin_abs(width, q_pos, gap) == 0.5
    assert ThetaPolicy(margin=0.5, margin_mode="relative_qpos").margin_abs(width, q_pos, gap) == 0.5 * 255.0 * 50.0


def test_negative_gap_is_clamped():
    """gap 이 음수면 θ 를 낮추게 되어 종단이 오히려 줄어든다. 0 으로 막아야 한다."""
    p = ThetaPolicy(margin=1.0, margin_mode="relative_gap")
    assert p.margin_abs(width=1.0, q_pos=1.0, gap=-5.0) == 0.0


def test_relative_qpos_hardcodes_255():
    """★ 알려진 한계 — relative_qpos 의 255 는 n_planes=8 을 가정한 상수다.

    ThetaPolicy 는 n_planes 를 갖고 있지 않아 스스로 알 방법이 없다.
    n_planes 를 바꾸면 이 모드만 조용히 틀린다.  (ARCHITECTURE.md 미결정 항목)
    """
    assert remaining_scale(0, N_PLANES) == 255
    got = ThetaPolicy(margin=1.0, margin_mode="relative_qpos").margin_abs(0.0, 1.0, 0.0)
    assert got == 255.0
    # 평면이 4개라면 R_0 = 15 여야 하는데 여전히 255 를 쓴다
    assert remaining_scale(0, 4) == 15
    assert got != float(remaining_scale(0, 4))


def test_unknown_names_are_rejected_loudly():
    for bad in ({"name": "no_such_policy"}, {"margin_mode": "no_such_mode"}):
        try:
            ThetaPolicy(**bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} 가 조용히 통과했다")


# ---------------------------------------------------------------------------
# 3. ThetaTracker — 정책별 갱신 시점
# ---------------------------------------------------------------------------
_LOWER = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
_LIVE = np.ones(5, dtype=bool)


def test_every_plane_updates_every_time():
    t = ThetaTracker(ThetaPolicy(name="every_plane", top_k=2))
    assert t.update(_LOWER, _LIVE, m=1) == 90.0
    assert t.update(_LOWER + 5.0, _LIVE, m=2) == 95.0      # 갱신된다
    assert t.theta == 95.0


def test_once_at_m_freezes_at_the_right_plane():
    t = ThetaTracker(ThetaPolicy(name="once_at_m", top_k=2, once_at_m=3))
    assert t.update(_LOWER, _LIVE, m=1) == -np.inf         # 아직 확정 전
    assert t.update(_LOWER, _LIVE, m=2) == -np.inf
    assert t.update(_LOWER, _LIVE, m=3) == 90.0            # 여기서 확정
    assert t.update(_LOWER + 100.0, _LIVE, m=4) == 90.0    # 이후 고정


def test_once_at_m_beyond_last_plane_never_fires():
    """★ 경계 결함 — once_at_m 이 n_planes-1 보다 크면 θ 가 영원히 -inf 다.

    run_step 의 루프가 m >= n_planes 에서 continue 하므로 update(m=8) 이
    호출되지 않는다. 그러면 종단이 한 번도 일어나지 않는데 경고가 없다.
    sweeps.yaml 은 [1,2,3,4] 라 지금은 안 걸리지만, 8 을 넣으면 조용히 무종단이 된다.
    """
    t = ThetaTracker(ThetaPolicy(name="once_at_m", top_k=2, once_at_m=N_PLANES))
    for m in range(1, N_PLANES):        # run_step 이 실제로 도는 범위
        t.update(_LOWER, _LIVE, m=m)
    assert t.theta == -np.inf, "이 값이 유한해지면 경계 동작이 바뀐 것이다"


def test_prev_step_and_oracle_are_frozen_from_the_start():
    p = ThetaPolicy(name="prev_step", top_k=2)
    t = ThetaTracker(p, prev_theta=42.0)
    assert t.theta == 42.0
    assert t.update(_LOWER, _LIVE, m=1) == 42.0            # 이번 스텝 하한을 안 본다

    o = ThetaTracker(ThetaPolicy(name="oracle_fixed", top_k=2), oracle_theta=77.0)
    assert o.update(_LOWER, _LIVE, m=1) == 77.0


def test_prev_step_without_a_previous_theta_disables_termination():
    """첫 디코드 스텝에는 직전 θ 가 없다 -> -inf 로 굳어 종단이 안 일어난다."""
    t = ThetaTracker(ThetaPolicy(name="prev_step", top_k=2), prev_theta=None)
    assert t.update(_LOWER, _LIVE, m=1) == -np.inf


def test_every_policy_name_is_constructible():
    for name in POLICIES:
        ThetaTracker(ThetaPolicy(name=name, top_k=2), prev_theta=1.0, oracle_theta=1.0)


# ---------------------------------------------------------------------------
# 4. topk_indices
# ---------------------------------------------------------------------------
def test_topk_indices_is_sorted_descending():
    s = np.array([1.0, 9.0, 5.0, 7.0])
    np.testing.assert_array_equal(topk_indices(s, 3), np.array([1, 3, 2]))


def test_topk_indices_clamps_k():
    s = np.array([1.0, 9.0])
    assert topk_indices(s, 10).size == 2
    assert topk_indices(s, 0).size == 0


def test_topk_indices_excludes_neg_inf():
    """★ 2026-08-28 결정 — `-inf` 를 걸러낸다. 반환 길이가 k 보다 짧을 수 있다.

    종단된 토큰은 masked_scores() 가 -inf 로 채운다. 예전에는 그것들도 후보에
    들어가서, 생존 토큰이 k 개보다 적으면 argpartition 이 -inf 중 아무거나 골라
    자리를 메웠다. **top-k 보존율이 부풀려졌다.**

        K_TOP=16 기준 실측    eval_k=32 : 0.7239 -> 0.6285
                             eval_k=64 : 0.7407 -> 0.3691   두 배 부풀려져 있었다

    `eval_k <= K_TOP` 이면 가드가 생존자를 K_TOP 개 이상 보장하므로 차이가 없다.
    그래서 지금 수치는 안 바뀐다. 조건이 바뀔 때 조용히 틀리는 것을 막는 수정이다.
    """
    s = np.array([5.0, -np.inf, -np.inf, -np.inf])
    idx = topk_indices(s, 3)
    assert idx.size == 1, "유한한 값이 하나뿐이면 하나만 나와야 한다"
    assert idx[0] == 0
    assert np.all(np.isfinite(s[idx]))

    # 전부 -inf 면 빈 배열
    assert topk_indices(np.full(4, -np.inf), 2).size == 0

    # 유한한 값이 충분하면 예전과 같다
    s2 = np.array([1.0, 9.0, 5.0, 7.0])
    np.testing.assert_array_equal(topk_indices(s2, 3), np.array([1, 3, 2]))


def test_topk_indices_callers_tolerate_short_returns():
    """★ 반환이 k 보다 짧아질 수 있으니 호출부 3곳이 감당하는지 확인한다.

      terminator.py:122   후보를 live 로 걸러 k <= 생존 수 -> 항상 k 개
      decode_loop.py:228  참 점수라 전부 유한 -> 항상 k 개
      decode_loop.py:229  종단 토큰이 -inf -> 짧아질 수 있다. intersect1d 는 무관
    """
    from src.bounds import step_bounds
    from src.masked_sum import partial_dots
    from src.quantize import to_bitplanes
    from src.terminator import masked_scores, run_step

    rng = np.random.default_rng(0)
    K = rng.integers(0, 256, (30, 8))
    q = rng.integers(-127, 128, 8)
    p = partial_dots(q, to_bitplanes(K, N_PLANES))[:, 0, :]
    r = run_step(p, step_bounds(q, N_PLANES), ThetaPolicy(name="every_plane", top_k=4))

    # 가드 경로 — 항상 정확히 k 개
    live_upper = np.where(r.alive, 1.0, -np.inf)
    assert topk_indices(live_upper, min(4, int(r.alive.sum()))).size == min(4, int(r.alive.sum()))

    # 보존율 경로 — 생존자보다 큰 k 를 물으면 짧아진다
    ms = masked_scores(r)
    n_alive = int(r.alive.sum())
    got = topk_indices(ms, n_alive + 5)
    assert got.size == n_alive, "생존자 수만큼만 나와야 한다"
    assert np.intersect1d(got, np.flatnonzero(r.alive)).size == n_alive
