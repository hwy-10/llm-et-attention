"""출력 경로 (`OUTPUT` 단계) — ARCHITECTURE.md 7.1  ★ 팀 2 소유

상태 기계의 마지막 단계다. 생존 집합을 정하는 것이 `term_ctrl`(팀 2 블록)이므로
이 단계도 팀 2 가 맡는다.

    생존 토큰마다 (인덱스, 점수) 한 쌍을 스트림으로 낸다

      score_idx    [IDX_BITS-1 : 0]           토큰 번호
      score_data   [DATA_BITS-1 : 0] signed   zero-point 보정 후 정수 점수
      score_valid                             1이면 이번 사이클 값이 유효
      score_last                              마지막 토큰에서 1

★ `score_last` 가 왜 필요한가 ★
    내보내는 개수가 **가변**이다. 정확 모드의 생존 집합은 top-k 의 상위집합이라
    항상 K_TOP 개 이상이고 스텝마다 다르다. 다운스트림(softmax)이 끝을 알 방법이
    없으면 스트림을 끊을 수 없다.

★★ 상한이 여기서 참값으로 무너진다 ★★

    U_m = S_m + (2^(n-m) - 1) · Q+       평면을 n 개 다 처리하면 m = n 이므로
    U_n = S_n + 0 · Q+ = S_n             잔여 계수가 0 이 된다

    즉 **출력 시점에는 상한 = 확정 점수**다. 7.1.4 의 "상한 U_m 이 큰 순으로 자른다"는
    실제로는 "점수가 큰 순으로 자른다"와 같은 말이고, 그래서 잘라내도 참 top-K_TOP 이
    남는 것이 실측이 아니라 **증명**이 된다 (OUT_BUF >= K_TOP 이기만 하면 된다).
    -> test_output_stage.py::test_truncation_preserves_true_topk

★★ 출력 버퍼는 따로 두지 않는다 — 2026-08-29 정정 ★★

    8/28 사양은 OUT_BUF = 2 x K_TOP = 32 였고 "실측 최대 31" 이 근거였다.
    이 모듈로 다시 재니 **시드에 따라 34 까지 나온다.** 32 로 자르면 그 스텝은
    무손실이 아니다.

    그런데 애초에 버퍼가 필요 없다. 내보낼 (토큰 번호, 확정 점수) 는 둘 다
    데이터패스에 이미 있다 — S_m 레지스터 파일(N_MAX x 22b, 평면 누산에 필수)과
    active 마스크(N_MAX x 1b, term_ctrl 이 쓰는 그 비트). OUTPUT 은 active 를
    훑으며 S_m 을 읽어 내보내면 되고, 새 저장소가 0 이다.

        2 x K_TOP 버퍼안 :  +1,056b  + 부분정렬 회로  + 무손실 깨짐
        S_m 직접 스캔    :      +0b  + 스캔 카운터    + 무손실 보장

    `out_buf` 는 그래서 "자르는 지점" 이 아니라 **넘으면 안 되는 상한(= N_MAX)** 이고,
    CNT_TRUNC 가 0 으로 유지되는지가 검증 항목이 된다. 자르기 경로는 남겨 두되
    (버퍼를 줄이는 변형을 재려면 필요하다) 채택 설정에서는 절대 타지 않는다.

★ zero-point 보정은 여기서 한 번만 ★

    s_real = scale_q · ( S_n(j) - Σ_i q_st_i · z_i ) / sqrt(d)
                                 └─ 스텝당 상수 ─┘

    모든 토큰에 같은 상수라 순위에 영향이 없다. 그래서 `term_ctrl` 은 정수 점수만
    보고 판정하고, 보정은 여기서 24b 감산기 하나로 끝낸다.
    `scale_q` 와 `/sqrt(d)` 는 실수 배율이라 하드웨어 밖으로 넘긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import apply_overrides, read_fields
from .terminator import StepResult

# 항목 이름 -> config/hardware.yaml 경로.  schedule.BRAM_WIRING 과 같은 형식.
OUTPUT_WIRING = (
    ("out_buf",   "hardware.output.out_buf",         int),
    ("idx_bits",  "hardware.output.idx_bits",        int),
    ("data_bits", "hardware.output.score_data_bits", int),
)


@dataclass(frozen=True)
class OutputSpec:
    """출력 경로 구성. -> ARCHITECTURE.md 7.1.2 / 7.1.4"""

    # ★ 기본값은 config/hardware.yaml 과 일부러 다르게 둔다 (memory.BramSpec 과 같은 이유).
    #   둘이 같으면 배선이 끊겨도 숫자가 안 변해 알 수 없다.
    #
    # out_buf 는 "여기서 자른다" 가 아니라 **넘으면 안 되는 상한** 이다.
    # 채택값 = N_MAX (512). -> config/hardware.yaml output 절의 정정 기록
    out_buf: int = 8
    idx_bits: int = 9           # log2(N_MAX = 512)
    data_bits: int = 24         # 3.1 절 비트폭 표

    def capacity_ok(self, top_k: int) -> bool:
        """가드의 하한을 담을 수 있는가.

        가드가 최소 top_k 개를 남기므로 out_buf < top_k 면 **매 스텝 자른다.**
        그건 버퍼가 아니라 설계 오류다.
        """
        return self.out_buf >= top_k

    def is_lossless(self, n_max: int) -> bool:
        """자르기가 원리적으로 불가능한가.

        이론적 최악은 활성 토큰 전체다 — θ 가 미확정인 디코드 초반
        (활성 < K_TOP) 에는 아무도 죽지 않는다. out_buf >= N_MAX 면
        그 최악조차 담기므로 자르기가 일어날 수 없다.
        """
        return self.out_buf >= n_max


@dataclass
class OutputStream:
    """한 스텝이 내보낸 스트림. 인덱스 순(오름차순)이며 하드웨어 방출 순서와 같다."""

    idx: np.ndarray             # (n_emit,) int  — score_idx
    data: np.ndarray            # (n_emit,) int  — score_data (zero-point 보정 후)
    truncated: int = 0          # 이번 스텝에서 버린 생존 토큰 수. 0 이면 무손실
    n_alive: int = 0            # 자르기 전 생존 수

    @property
    def n_emit(self) -> int:
        return int(self.idx.size)

    @property
    def last(self) -> np.ndarray:
        """score_last — 마지막 유효 토큰에서만 1.

        비어 있으면 전부 0 이다. 그 경우 다운스트림은 `score_valid` 가 한 번도
        서지 않는 것으로 빈 스텝을 안다.
        """
        f = np.zeros(self.n_emit, dtype=bool)
        if self.n_emit:
            f[-1] = True
        return f

    def check_widths(self, spec: OutputSpec) -> None:
        """비트폭 초과를 조용히 넘기지 않는다. 골든모델은 int64 라 그냥 담긴다."""
        if self.n_emit == 0:
            return
        if int(self.idx.max()) >= (1 << spec.idx_bits):
            raise ValueError(
                f"score_idx {int(self.idx.max())} 가 {spec.idx_bits}b 를 넘는다 "
                f"(N_MAX 를 키웠으면 idx_bits 도 키워야 한다)"
            )
        lim = 1 << (spec.data_bits - 1)
        if int(np.abs(self.data).max()) >= lim:
            raise ValueError(
                f"score_data |{int(np.abs(self.data).max())}| 가 signed {spec.data_bits}b "
                f"[{-lim}, {lim - 1}] 를 넘는다"
            )


@dataclass
class OutputCounters:
    """제어/상태 레지스터. -> ARCHITECTURE.md 8.2"""

    cnt_trunc: int = 0          # 자르기가 일어난 **스텝** 수
    tokens_dropped: int = 0     # 버린 토큰 총수
    max_alive: int = 0          # 관측된 최대 생존 수 — OUT_BUF 근거가 된다
    hist: dict = field(default_factory=dict)   # 생존 수 -> 스텝 수

    def observe(self, stream: OutputStream) -> None:
        self.max_alive = max(self.max_alive, stream.n_alive)
        self.hist[stream.n_alive] = self.hist.get(stream.n_alive, 0) + 1
        if stream.truncated:
            self.cnt_trunc += 1
            self.tokens_dropped += stream.truncated


def emit_step(
    result: StepResult,
    spec: OutputSpec,
    *,
    zp_correction: int = 0,
    counters: OutputCounters | None = None,
) -> OutputStream:
    """생존 집합을 (인덱스, 점수) 스트림으로 내보낸다.

    Parameters
    ----------
    result : `terminator.run_step` 의 결과
    spec   : 출력 경로 구성
    zp_correction : Σ_i q_st_i · z_i  — 스텝당 상수. `accumulator.FoldedQuery` 가 낸다
    counters : 주면 CNT_TRUNC 등을 누적한다

    Notes
    -----
    자르기는 **점수 내림차순**으로 한다. 위 독스트링대로 출력 시점에는 상한이
    확정 점수와 같으므로 7.1.4 의 "상한 상위" 와 같은 동작이다.
    동점은 인덱스가 작은 쪽을 남긴다 (`np.argsort(kind="stable")`).
    """
    alive = np.asarray(result.alive, dtype=bool)
    if alive.ndim != 1:
        raise ValueError(f"result.alive must be 1-D, got {alive.shape}")
    if spec.out_buf <= 0:
        raise ValueError(f"out_buf = {spec.out_buf}: must be >= 1")

    idx = np.flatnonzero(alive)
    n_alive = int(idx.size)

    truncated = 0
    if n_alive > spec.out_buf:
        # 점수 내림차순 상위 out_buf 개. 안정 정렬이라 동점은 인덱스 순.
        order = np.argsort(-result.s_int[idx].astype(np.int64), kind="stable")
        idx = np.sort(idx[order[: spec.out_buf]])     # 방출은 인덱스 순
        truncated = n_alive - spec.out_buf

    data = result.s_int[idx].astype(np.int64) - int(zp_correction)
    stream = OutputStream(idx=idx, data=data, truncated=truncated, n_alive=n_alive)
    stream.check_widths(spec)
    if counters is not None:
        counters.observe(stream)
    return stream


def output_spec_from_config(cfg, **overrides) -> OutputSpec:
    """config/hardware.yaml -> OutputSpec."""

    # 설정 파싱 (누락 항목: 기본값 대체 + ConfigDefaultWarning)
    fields = read_fields(cfg, OutputSpec, OUTPUT_WIRING)

    return apply_overrides(OutputSpec(**fields), overrides)
