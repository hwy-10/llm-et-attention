"""★ 디코드 루프 시뮬레이터 ★ — 배경지식 가이드 3.2 / 8.3 1단계

T 가 자라는 실제 디코딩을 그대로 돈다.

    step s:  쿼리 q_{warmup+s} 가 토큰 0 .. warmup+s-1 을 본다
             -> 활성 토큰 수가 스텝마다 늘어난다

"평균 종단 시점"은 T 에 강하게 의존하므로 고정 스냅샷으로는 답이 안 나온다.
그래서 루프를 돈다.

속도 설계
--------
부분 내적 P_b 는 margin / θ정책 / top-k 와 무관하다.
따라서 **한 번만** 계산해 두고(Workbench) 모든 스윕이 재사용한다.
루프 자체는 스텝당 numpy 연산 몇 개뿐이라 512스텝이 순식간에 끝난다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .accumulator import FoldedQuery, exact_int_scores, fold_and_quantize_query, to_real_scores
from .bounds import StepBounds, batch_step_bounds
from .dataset import QKSnapshot
from .designs import DESIGNS, DesignResult, run_design
from .masked_sum import partial_dots
from .memory import BramSpec, ReadAccount, _ceil_div
from .quantize import KeyQuant, quantize_key, to_bitplanes
from .schedule import ScheduleSpec
from .threshold import topk_indices


# ---------------------------------------------------------------------------
@dataclass
class DecodeWorkbench:
    """스윕 전체가 공유하는 전처리 결과."""

    key: KeyQuant
    fq: FoldedQuery
    partials: np.ndarray          # (n_planes, n_steps, n_tokens) int32
    exact_int: np.ndarray         # (n_steps, n_tokens) 참 정수 점수
    step_tokens: np.ndarray       # (n_steps,) 각 스텝의 활성 토큰 수
    q_pos: np.ndarray             # (n_steps,)
    q_neg: np.ndarray             # (n_steps,)
    head_dim: int
    n_planes: int
    warmup: int
    build_seconds: float = 0.0

    @property
    def n_steps(self) -> int:
        return self.partials.shape[1]

    @classmethod
    def build(
        cls,
        snap: QKSnapshot,
        *,
        n_planes: int = 8,
        warmup: int = 32,
        seq_len: int | None = None,
        step_stride: int = 1,
        key_granularity: str = "per_channel",
        clip_percentile: float = 100.0,
        chunk: int = 0,
    ) -> "DecodeWorkbench":
        t0 = time.perf_counter()
        seq_len = int(seq_len or snap.n_tokens)
        seq_len = min(seq_len, snap.n_tokens, snap.q.shape[0])
        warmup = int(np.clip(warmup, 1, seq_len - 1))

        key = quantize_key(
            snap.k[:seq_len], bits=n_planes,
            granularity=key_granularity, clip_percentile=clip_percentile,
        )
        planes = to_bitplanes(key.stored, n_planes)          # (P, T, d)

        step_idx = np.arange(warmup, seq_len, max(1, int(step_stride)), dtype=np.int64)
        fq = fold_and_quantize_query(snap.q[step_idx], key)

        partials = partial_dots(fq.stored, planes, chunk=chunk)   # (P, S, T)
        exact = exact_int_scores(fq.stored, key)                   # (S, T)
        q_pos, q_neg = batch_step_bounds(fq.stored, n_planes)

        return cls(
            key=key, fq=fq, partials=partials.astype(np.int32),
            exact_int=exact, step_tokens=step_idx.astype(np.int64),
            q_pos=q_pos, q_neg=q_neg,
            head_dim=snap.head_dim, n_planes=n_planes, warmup=warmup,
            build_seconds=time.perf_counter() - t0,
        )

    def bounds_at(self, s: int) -> StepBounds:
        return StepBounds(int(self.q_pos[s]), int(self.q_neg[s]), self.n_planes)

    def real_scores(self, s: int, s_int: np.ndarray) -> np.ndarray:
        """정수 점수 -> 어텐션 로짓 (zero-point 보정 + 1/sqrt(d))."""
        finite = np.isfinite(s_int)
        out = np.full(s_int.shape, -np.inf, dtype=np.float64)
        vals = (s_int[finite] - float(self.fq.zp_correction[s])) * float(
            np.asarray(self.fq.scale).reshape(-1)[s]
        )
        out[finite] = vals / np.sqrt(self.head_dim)
        return out


# ---------------------------------------------------------------------------
@dataclass
class DecodeResult:
    """디코드 루프 한 번의 결과."""

    config: dict
    summary: dict
    per_step: dict[str, np.ndarray] = field(default_factory=dict)

    def record(self) -> dict:
        """outputs/raw 에 저장할 평탄한 레코드 한 줄."""
        rec = dict(self.config)
        rec.update(self.summary)
        return rec


def _softmax(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    if not np.any(finite):
        return out
    z = x[finite] - np.max(x[finite])
    e = np.exp(z)
    out[finite] = e / e.sum()
    return out


def run_decode(
    wb: DecodeWorkbench,
    *,
    design: str = "exact",
    top_k: int = 8,
    margin: float = 0.0,
    margin_mode: str = "relative_width",
    theta_policy: str = "every_plane",
    once_at_m: int = 3,
    schedule_policy: str | None = None,
    sched: ScheduleSpec | None = None,
    bram: BramSpec | None = None,
    eval_top_k: tuple[int, ...] = (1, 4, 8, 16),
    keep_trace: bool = True,
) -> DecodeResult:
    """디코드 루프를 끝까지 돌며 정확도와 하드웨어 회계를 누적한다."""
    if design not in DESIGNS:
        raise KeyError(f"unknown design {design!r}")
    sched = sched or ScheduleSpec()
    bram = bram or BramSpec()

    n_steps = wb.n_steps
    total_cycles = 0            # 연산 사이클 (기존 의미 — 절대 바꾸지 않는다)
    total_memory_cycles = 0     # BRAM 포트로 나눈 메모리 사이클
    total_cycles_mem = 0        # 둘을 합친 값 (sched.mem_overlap 가정)
    total_baseline_cycles = 0
    total_baseline_cycles_mem = 0   # 기준선도 메모리를 포함해 잰 값
    total_reads = ReadAccount()
    term_planes: list[float] = []
    survivor_fracs: list[float] = []
    retention = {k: [] for k in eval_top_k}
    kl_list: list[float] = []
    kl_oracle_list: list[float] = []
    cycles_trace = np.zeros(n_steps, dtype=np.int64)
    words_trace = np.zeros(n_steps, dtype=np.int64)
    term_trace = np.zeros(n_steps, dtype=np.float64)
    # 평면별 생존 곡선 — 종단이 몇 번째 평면부터 실제로 시작되는지 보여준다.
    # (2단계 처리의 분할점 m0 를 정하는 근거이기도 하다. 가이드 6.3-(2))
    plane_live = np.zeros(wb.n_planes, dtype=np.float64)
    plane_total = np.zeros(wb.n_planes, dtype=np.float64)

    from .schedule import baseline_cycles as _base_cyc
    from .schedule import dense_cycles as _dense_cyc

    prev_theta_real: float | None = None

    for s in range(n_steps):
        n_active = int(wb.step_tokens[s])
        p = wb.partials[:, s, :n_active]
        bounds = wb.bounds_at(s)
        scale_s = float(np.asarray(wb.fq.scale).reshape(-1)[s])
        zp_s = float(wb.fq.zp_correction[s])

        # 직전 스텝 θ 를 이번 스텝의 정수 도메인으로 환산
        prev_theta_int = None
        if prev_theta_real is not None and np.isfinite(prev_theta_real):
            prev_theta_int = prev_theta_real * np.sqrt(wb.head_dim) / scale_s + zp_s

        oracle_theta_int = None
        if theta_policy == "oracle_fixed":
            ex = wb.exact_int[s, :n_active]
            kk = min(top_k, n_active)
            oracle_theta_int = float(np.partition(ex, -kk)[-kk])

        res: DesignResult = run_design(
            design, p, bounds,
            top_k=top_k, margin=margin, margin_mode=margin_mode,
            theta_policy=theta_policy, once_at_m=once_at_m,
            schedule_policy=schedule_policy, sched=sched, bram=bram,
            prev_theta_int=prev_theta_int, oracle_theta_int=oracle_theta_int,
        )

        total_cycles += res.cycles
        if res.schedule is not None:
            total_memory_cycles += res.schedule.memory_cycles
            total_cycles_mem += res.schedule.total_cycles
        # 기준선도 같은 잣대로 재야 한다. 분자만 연산이면 기준선을 기준선으로
        # 나눠도 1.0 이 안 나온다 (실측 0.25). 기준선은 K 를 전부 읽으므로
        # 메모리 몫은 dense 워드다.
        base_cyc = _base_cyc(n_active, sched)
        base_mem = _ceil_div(res.reads.words_dense, bram.n_ports, what="n_ports")
        total_baseline_cycles += base_cyc
        total_baseline_cycles_mem += (max(base_cyc, base_mem) if sched.mem_overlap
                                      else base_cyc + base_mem)
        total_reads += res.reads
        term_planes.append(res.mean_term_plane)
        survivor_fracs.append(res.survivor_frac)
        cycles_trace[s] = res.cycles
        words_trace[s] = res.reads.words_bram
        term_trace[s] = res.mean_term_plane
        if res.step_result is not None:
            plane_live += res.step_result.read_live.sum(axis=1)
            plane_total += n_active

        # --- 정확도 ------------------------------------------------------
        # ★ 두 종류의 손실을 반드시 분리해서 본다 ★
        #   (a) 종단으로 인한 손실       -> top-k 보존율. 정확 모드는 1.0 이어야 한다.
        #   (b) top-k 희소화 자체의 손실 -> 종단과 무관. 오라클 기준선으로 잰다.
        # 정확 모드의 softmax_kl 이 0이 아닌 것은 (b) 때문이지 (a) 때문이 아니다.
        ref_int = wb.exact_int[s, :n_active].astype(np.float64)
        ref_real = wb.real_scores(s, ref_int)
        test_real = wb.real_scores(s, res.scores)
        for k in eval_top_k:
            kk = int(min(k, n_active))
            a = topk_indices(ref_real, kk)
            b = topk_indices(test_real, kk)
            retention[k].append(np.intersect1d(a, b).size / kk)

        pr, pt = _softmax(ref_real), _softmax(test_real)
        kl_list.append(float(np.sum(pr * (np.log(pr + 1e-12) - np.log(pt + 1e-12)))))

        # top-k 오라클: 참 top-k 만 남긴 희소 어텐션
        kk_o = int(min(top_k, n_active))
        oracle_real = np.full(n_active, -np.inf, dtype=np.float64)
        oi = topk_indices(ref_real, kk_o)
        oracle_real[oi] = ref_real[oi]
        po = _softmax(oracle_real)
        kl_oracle_list.append(
            float(np.sum(pr * (np.log(pr + 1e-12) - np.log(po + 1e-12))))
        )

        # 다음 스텝의 prev_step θ
        kk = int(min(top_k, n_active))
        finite_ref = ref_real[np.isfinite(ref_real)]
        if finite_ref.size >= kk:
            prev_theta_real = float(np.partition(finite_ref, -kk)[-kk])

    kl_design = float(np.mean(kl_list)) if kl_list else 0.0
    kl_oracle = float(np.mean(kl_oracle_list)) if kl_oracle_list else 0.0
    dense_cyc = sum(
        _dense_cyc(int(wb.step_tokens[s]), wb.n_planes, sched) for s in range(n_steps)
    )
    summary = {
        "n_steps": n_steps,
        "total_cycles": int(total_cycles),
        "baseline_cycles": int(total_baseline_cycles),
        "dense_cycles": int(dense_cyc),
        "cycle_speedup_vs_baseline": (
            total_baseline_cycles / total_cycles if total_cycles else 0.0
        ),
        # ★ 제안 설계의 진짜 비교 대상은 ②(종단 없는 순차 처리)다 (가이드 8.1절)
        "cycle_saving_vs_seq": 1.0 - total_cycles / dense_cyc if dense_cyc else 0.0,
        # ★ 메모리 수행 시간을 포함한 값 — 여기부터는 새로 추가된 항이다 ★
        # total_cycles 는 연산만 센다. Decode 는 메모리 병목이라는 것이 이 프로젝트의
        # 전제이므로, BRAM 포트로 나눈 메모리 사이클과 합친 값을 함께 낸다.
        # 기존 결론이 조용히 움직이지 않도록 total_cycles 는 그대로 두고 더해서 낸다.
        "total_memory_cycles": int(total_memory_cycles),
        "total_cycles_with_memory": int(total_cycles_mem),
        "memory_bound_frac": (
            total_memory_cycles / total_cycles if total_cycles else 0.0
        ),
        "cycle_speedup_vs_baseline_with_memory": (
            total_baseline_cycles_mem / total_cycles_mem if total_cycles_mem else 0.0
        ),
        "total_baseline_cycles_with_memory": int(total_baseline_cycles_mem),
        "mem_overlap": bool(sched.mem_overlap),
        "mean_term_plane": float(np.mean(term_planes)) if term_planes else 0.0,
        "mean_survivor_frac": float(np.mean(survivor_fracs)) if survivor_fracs else 1.0,
        # (b) 종단 + top-k 희소화를 합친 손실
        "softmax_kl": kl_design,
        # (b) 만: 참 top-k 만 남겼을 때의 손실 (종단과 무관한 기준선)
        "softmax_kl_topk_oracle": kl_oracle,
        # 종단이 오라클 대비 추가로 유발한 손실. 정확 모드에서는 0 이하여야 한다
        # (정확 모드는 top-k 보다 많은 토큰을 남기므로 오라클보다 오히려 정확하다)
        "kl_excess_from_termination": kl_design - kl_oracle,
    }
    summary.update(total_reads.as_dict())
    for k in eval_top_k:
        # k=1 은 utils/metrics.py 와 같은 이름으로 낸다. 같은 값이 두 이름으로
        # 나가면 예산 검사(top1_acc)와 실험 CSV(top1_retention)가 다른 열을 본다.
        key = "top1_acc" if k == 1 else f"top{k}_retention"
        summary[key] = float(np.mean(retention[k])) if retention[k] else 1.0
    # 평면별 생존 비율 (평면 0 = MSB). 종단이 몇 번째 평면부터 시작되는지.
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(plane_total > 0, plane_live / plane_total, 1.0)
    for t in range(wb.n_planes):
        summary[f"live_frac_p{t}"] = float(frac[t])
    first = np.flatnonzero(frac < 0.999)
    summary["first_terminating_plane"] = int(first[0]) if first.size else wb.n_planes

    cfg = {
        "design": design,
        "top_k": top_k,
        "margin": margin,
        "margin_mode": margin_mode,
        "theta_policy": theta_policy,
        "once_at_m": once_at_m,
        "schedule_policy": schedule_policy or "",
        "lanes": sched.lanes,
        "word_tokens": bram.word_tokens,
        "decision_latency_planes": bram.decision_latency_planes,
        "seq_len": int(wb.step_tokens[-1]) if wb.n_steps else 0,
        "warmup": wb.warmup,
    }
    per_step: dict[str, np.ndarray] = {}
    if keep_trace:
        per_step = {
            "n_active": wb.step_tokens.copy(),
            "cycles": cycles_trace,
            "words_bram": words_trace,
            "mean_term_plane": term_trace,
        }
    return DecodeResult(config=cfg, summary=summary, per_step=per_step)


def workbench_from_config(cfg, snap: QKSnapshot, **overrides) -> DecodeWorkbench:
    dec = cfg.get("model.decode", {}) or {}
    key_cfg = cfg.get("quant.key", {}) or {}
    kw = dict(
        n_planes=cfg.n_planes,
        warmup=int(dec.get("warmup_tokens", 32)),
        seq_len=int(dec.get("seq_len", 512)),
        step_stride=int(dec.get("step_stride", 1)),
        key_granularity=str(key_cfg.get("granularity", "per_channel")),
        clip_percentile=float(key_cfg.get("clip_percentile", 100.0)),
    )
    kw.update(overrides)
    return DecodeWorkbench.build(snap, **kw)
