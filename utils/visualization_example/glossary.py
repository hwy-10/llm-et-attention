"""프로젝트 용어 사전 — 같은 그림 위에 겹쳐 놓는다.

★ 용어를 따로 그리지 않는 이유 ★

이 프로젝트의 용어는 대부분 **같은 격자 위에 산다.**

    토큰      = 격자의 한 열
    비트평면  = 격자의 한 행
    레인      = 한 행의 가로 덩어리 (lanes 개)
    BRAM 워드 = 한 행의 가로 덩어리 (word_tokens 개)   <- 레인과 모양이 같다

용어마다 그림을 따로 그리면 이 **겹침**이 보이지 않는다. 그런데 겹침이야말로
헷갈리는 지점이다 — 레인과 워드는 둘 다 "행의 가로 덩어리"이고, 하필 실제
설정에서 둘 다 32라 구분이 안 간다. 같은 격자에 올려 두면 "지금은 같지만 다른
것"임이 눈에 보인다.

캔버스가 셋이다.

``grid``
    평면 × 토큰. 저장·읽기 계열 용어가 여기 산다.
``bracket``
    수직선. 점수·판정 계열(S_m, R_m, L_m, θ)이 여기 산다.
``vector``
    q 와 K[j] 의 head_dim 칸. 내적 계열이 여기 산다.

수치는 ``config/hardware.yaml`` 에서 읽는다 — 손으로 베끼면 설정이 바뀔 때
사전이 조용히 낡는다.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DEMO", "build"]

#: 화면에 그리는 축소판 크기. 실제 값은 build() 가 config 에서 읽어 함께 낸다.
#: ★ 축소해도 lanes == word_tokens 관계는 유지한다 — 그게 요점이기 때문이다.
DEMO = {
    "n_tokens": 16,
    "n_planes": 8,
    "lanes": 4,
    "word_tokens": 4,
    "head_dim": 8,
    "top_k": 3,
    "m": 3,          # bracket 캔버스에서 "m 장 처리한 시점"
}


def _real_spec() -> dict[str, Any]:
    """실제 설정값. 못 읽으면 dataclass 기본값으로 되돌아간다."""
    try:
        from src.config import load_config
        from src.schedule import bram_from_config, spec_from_config

        cfg = load_config()
        spec, bram = spec_from_config(cfg), bram_from_config(cfg)
        return {
            "lanes": spec.lanes,
            "word_tokens": bram.word_tokens,
            "word_bits": bram.word_bits,
            "n_ports": bram.n_ports,
            "batch_size": spec.batch_size,
            "decision_latency_planes": bram.decision_latency_planes,
            "decision_latency_mode": bram.decision_latency_mode,
            "pipeline_cycles": bram.pipeline_cycles,
            "n_planes": int(cfg.get("quant.planes.n_planes", 8) or 8),
            "head_dim": int(cfg.get("model.model.head_dim", 64) or 64),
            "source": "config/hardware.yaml",
        }
    except Exception:
        from src.memory import BramSpec
        from src.schedule import ScheduleSpec

        s, b = ScheduleSpec(), BramSpec()
        return {
            "lanes": s.lanes, "word_tokens": b.word_tokens, "word_bits": b.word_bits,
            "n_ports": b.n_ports, "batch_size": s.batch_size,
            "decision_latency_planes": b.decision_latency_planes,
            "decision_latency_mode": b.decision_latency_mode,
            "pipeline_cycles": b.pipeline_cycles,
            "n_planes": 8, "head_dim": 64,
            "source": "dataclass 기본값 (config 를 읽지 못함)",
        }


# ---------------------------------------------------------------------------
# 용어
# ---------------------------------------------------------------------------
# highlight 는 캔버스가 해석한다.
#   grid    : column / row / chunk / cell / all / live / dead / rows_done / rows_left
#   bracket : marker (l | s_m | upper | theta | truth | gap)
#   vector  : q | k | pair | bit

_TERMS: list[dict[str, Any]] = [
    # ---------------- 저장 · 데이터 ----------------
    {
        "id": "token", "term": "토큰", "en": "token", "group": "데이터",
        "one": "KV 캐시에 쌓인 과거 단어 하나. **격자의 한 열.**",
        "canvas": "grid", "highlight": {"kind": "column", "index": 5},
        "real": "T (문맥 길이). 목표 512까지",
        "detail": (
            "디코드 스텝마다 **하나씩 늘어난다.** 토큰 t 의 K 는 t 스텝에 한 번 저장되고 "
            "이후 모든 스텝에서 읽힌다 — 쓰기 1 : 읽기 N 이라 병목이 연산이 아니라 메모리다.\n\n"
            "코드에서는 `n_tokens` / `n_active` 로 나온다."
        ),
        "confuse": ["word"],
    },
    {
        "id": "head_dim", "term": "head 차원", "en": "head_dim, d", "group": "데이터",
        "one": "q 와 K 한 줄의 길이. 내적을 몇 개 더하는가.",
        "canvas": "vector", "highlight": {"kind": "pair"},
        "real": "64",
        "detail": (
            "Llama 3.2 1B 의 head_dim 은 64 다. 부분 내적은 이 64개를 더하는 것이고, "
            "그래서 가산 트리가 **6단 63개**가 된다 (`log2(64)=6`).\n\n"
            "격자 그림에는 나오지 않는다 — 격자의 칸 하나가 이미 '토큰 하나의 한 평면'이고, "
            "그 안에 head_dim 개의 비트가 또 들어 있기 때문이다."
        ),
    },
    {
        "id": "plane", "term": "비트평면", "en": "bit-plane", "group": "데이터",
        "one": "K 를 자릿수별로 쪼갠 한 장. **격자의 한 행.**",
        "canvas": "grid", "highlight": {"kind": "row", "index": 2},
        "real": "8장 (INT8)",
        "detail": (
            "INT8 Key 를 8장으로 쪼갠다. 값은 전혀 바뀌지 않고 **메모리에 배치되는 방식만** 달라진다.\n\n"
            "```\n200 = 1 1 0 0 1 0 0 0\n       ↑\n       b7 평면\n```\n\n"
            "MSB(b7)부터 처리하므로 위쪽 행이 먼저다. 코드에서 `to_bitplanes()` 의 "
            "**index 0 이 b7** 이다."
        ),
        "confuse": ["cycle"],
    },
    {
        "id": "plane_weight", "term": "평면 자리값", "en": "plane weight", "group": "데이터",
        "one": "평면 b 의 무게. `[128, 64, 32, 16, 8, 4, 2, 1]` — **전부 양수.**",
        "canvas": "grid", "highlight": {"kind": "row", "index": 0},
        "real": "2^(7-b)",
        "detail": (
            "★ **전부 양수라는 것이 상한식의 전제다.** K 를 unsigned 로 저장하기 때문이다.\n\n"
            "다만 이건 **하드웨어 단순화이지 수학적 필연이 아니다** — MSB-first 라 "
            "부호 비트는 라운드 0에 확정되어 이미 '결정된 부분합'에 들어간다. "
            "signed 로 해도 상한식은 세울 수 있고, 다만 `R_m` 식이 평면마다 달라져 회로가 복잡해진다."
        ),
    },
    # ---------------- 계산 ----------------
    {
        "id": "partial", "term": "부분합", "en": "P_b, partial dot", "group": "계산",
        "one": "평면 b 하나에서 나온 내적. **곱셈 없이** 비트가 1인 자리의 q 를 더한 것.",
        "canvas": "vector", "highlight": {"kind": "bit"},
        "real": "partial_dots() 의 출력",
        "detail": (
            "```\nq = [5, -3, 7, 2]\nb7 평면 = [1, 0, 1, 0]\nP = 5 + 7 = 12\n```\n\n"
            "비트가 1인 자리의 q 만 더하므로 **곱셈기가 필요 없다.** 이것이 DSP 미사용 논거다.\n\n"
            "⚠ NumPy 코드는 `einsum` 으로 쓰여 있어 형식적으로는 곱셈이다. "
            "실제 DSP 0 은 RTL 합성 결과로 확인해야 한다."
        ),
    },
    {
        "id": "s_m", "term": "확정 부분합", "en": "S_m", "group": "계산",
        "one": "m 장까지 처리해서 **이미 확정된** 점수.",
        "canvas": "bracket", "highlight": {"kind": "marker", "which": "s_m"},
        "real": "Σ 2^(8-b) · P_b",
        "detail": (
            "평면을 하나 처리할 때마다 자리값을 곱해 더한다. **더 이상 바뀌지 않는 부분**이다.\n\n"
            "```\nS_1 = 128 x P_b7\nS_2 = S_1 + 64 x P_b6\n```\n\n"
            "격자로 보면 **이미 읽은 위쪽 행들**에서 나온 값이다."
        ),
    },
    {
        "id": "qplus", "term": "Q+ / Q−", "en": "Q_pos, Q_neg", "group": "계산",
        "one": "q 의 양수 합과 음수 합. **스텝당 한 번만** 계산한다.",
        "canvas": "vector", "highlight": {"kind": "q"},
        "real": "step_bounds() 의 출력",
        "detail": (
            "```\nq  = [5, -3, 7, 2]\nQ+ = 5 + 7 + 2 = 14\nQ− =        -3 = -3\n```\n\n"
            "★ **매 평면 다시 계산하면 회로 논거가 무너진다.** q 는 스텝 내내 고정이므로 "
            "한 번만 구해 레지스터에 넣어 두면 된다. 그래서 블록도에 "
            "`Q+ / Q− 레지스터` 가 따로 있다."
        ),
    },
    {
        "id": "r_m", "term": "남은 상한", "en": "R_m", "group": "계산",
        "one": "아직 안 읽은 평면이 **더할 수 있는 최대**. `(2^(8-m) − 1) · Q+`",
        "canvas": "bracket", "highlight": {"kind": "marker", "which": "upper"},
        "real": "bounds.r(m)",
        "detail": (
            "남은 비트가 **전부 1** 이면 최대다. 남은 자리값의 합이 `2^(8-m) − 1` 이므로 "
            "거기에 `Q+` 를 곱한다.\n\n"
            "★ **아직 읽지 않은 비트에 대해 아무 가정도 하지 않는다.** 추정이 아니라 증명이다 — "
            "이게 '무손실'의 근거다."
        ),
    },
    {
        "id": "bracket", "term": "구간", "en": "bracket, L_m ≤ s ≤ S_m + R_m", "group": "계산",
        "one": "참값이 반드시 들어 있는 범위. 평면을 처리할수록 **좁아진다.**",
        "canvas": "bracket", "highlight": {"kind": "marker", "which": "gap"},
        "real": "verify_bracket()",
        "detail": (
            "```\nL_m = S_m + (2^(8-m) − 1) · Q−     ← 최소\nU_m = S_m + (2^(8-m) − 1) · Q+     ← 최대\n```\n\n"
            "★ **이 프로젝트의 핵심 불변식이다.** `tests/test_bounds.py` 가 모든 평면·모든 토큰에 "
            "대해 이걸 강제한다. 여기가 깨지면 프로젝트 전체가 폐기된다."
        ),
    },
    # ---------------- 판정 ----------------
    {
        "id": "theta", "term": "문턱", "en": "θ, theta", "group": "판정",
        "one": "이 아래로 떨어지면 버린다. **활성 토큰 하한 중 k번째로 큰 값.**",
        "canvas": "bracket", "highlight": {"kind": "marker", "which": "theta"},
        "real": "kth_largest(L_m, top_k)",
        "detail": (
            "θ 를 '하한 중 k등'으로 잡으면 **무손실이 보장된다.** 상한이 θ 아래인 토큰은 "
            "남은 비트를 전부 1로 채워도 k등을 넘을 수 없기 때문이다.\n\n"
            "언제 확정하느냐가 정책이다 — `every_plane`(매번) / `once_at_m`(한 번) / "
            "`prev_step`(이전 스텝 값 재사용, **무손실 아님**) / `oracle_fixed`(참조선)."
        ),
    },
    {
        "id": "topk", "term": "상위 k", "en": "top-k", "group": "판정",
        "one": "살려야 하는 개수. **이 프로젝트의 유일한 종단 기준이다.**",
        "canvas": "bracket", "highlight": {"kind": "marker", "which": "theta"},
        "real": "top_k = 8 (실험 기본값)",
        "detail": (
            "θ 정책 4종이 전부 `kth_largest()` 를 쓴다. 즉 **상위 k 후보만이 종단 기준**이고, "
            "'점수가 충분히 작으면 버린다' 같은 절대 문턱은 없다.\n\n"
            "종단 뒤 어텐션은 살아남은 토큰만으로 재정규화한다."
        ),
    },
    {
        "id": "terminate", "term": "종단", "en": "early termination", "group": "판정",
        "one": "이 토큰은 더 안 읽기로 **확정.** 격자에서 그 아래 칸이 꺼진다.",
        "canvas": "grid", "highlight": {"kind": "dead"},
        "real": "term_plane[i]",
        "detail": (
            "상한이 θ 아래로 떨어진 순간 확정된다. **되살아나지 않는다** — "
            "누산이 동결(`frozen`)되고 이후 평면은 읽지 않는다.\n\n"
            "격자에서 한 토큰(열)의 아래쪽이 통째로 꺼지는 모양이 된다."
        ),
    },
    {
        "id": "read_live", "term": "읽기 마스크", "en": "read_live", "group": "판정",
        "one": "★ **두 팀의 경계.** (평면 × 토큰) bool — 격자 그 자체.",
        "canvas": "grid", "highlight": {"kind": "all"},
        "real": "(n_planes, n_tokens) bool",
        "detail": (
            "종단 팀(`terminator.run_step`)이 만들고 비트평면 팀"
            "(`memory.account_step` / `schedule.apply`)이 소비한다. "
            "**블록도의 붉은 화살표가 코드에서는 이 배열 하나다.**\n\n"
            "⚠ 단위 주의 — `read_live` 는 **토큰 마스크**이고 "
            "`account_step()` 출력은 **BRAM 워드 수**다. 섞이면 모든 절감 수치가 무의미해진다."
        ),
        "confuse": ["word"],
    },
    {
        "id": "latency", "term": "판정 지연", "en": "decision_latency", "group": "판정",
        "one": "판정이 읽기에 반영되기까지 걸리는 평면 수.",
        "canvas": "grid", "highlight": {"kind": "cell", "row": 4, "col": 5},
        "real": "1 (config)",
        "detail": (
            "평면 m 의 판정은 m 의 가산 트리가 끝나야 나온다. 그 시점에 평면 m+1 의 "
            "읽기 요청은 **이미 나갔을 수 있다.**\n\n"
            "```\nread_live[t] = live_history[t − latency]\n```\n\n"
            "★ `latency = 0` 으로 두면 절감이 과대평가된다. 실측: 0 → 3 으로 바꾸면 "
            "읽는 양이 1,360 → 2,012 로 늘어난다."
        ),
    },
    {
        "id": "margin", "term": "마진", "en": "margin", "group": "판정",
        "one": "일부러 손실을 감수해 더 아끼는 손잡이. **0이면 무손실.**",
        "canvas": "bracket", "highlight": {"kind": "marker", "which": "theta"},
        "real": "0.0 (정확 모드) ~ 0.25",
        "detail": (
            "θ 를 조금 올려 더 많이 버린다. 실측(합성 데이터, 시드 8개 평균):\n\n"
            "```\nmargin  top8 보존   words_bram\n  0.00     1.0000        51.1\n"
            "  0.05     0.9062        45.4\n  0.15     0.8594        39.9\n  0.25     0.7188        36.0\n```\n\n"
            "★ **손실을 감수하는 유일한 축이다.** 스케줄 정책은 점수를 전혀 바꾸지 않는다."
        ),
    },
    # ---------------- 하드웨어 ----------------
    {
        "id": "lane", "term": "레인", "en": "lane", "group": "하드웨어",
        "one": "한 사이클에 동시에 처리하는 토큰 수. **한 행의 가로 덩어리.**",
        "canvas": "grid", "highlight": {"kind": "chunk", "row": 2, "start": 0, "unit": "lanes"},
        "real": "32",
        "detail": (
            "평면 하나를 끝내려면 `ceil(토큰 수 / lanes)` 사이클이 걸린다.\n\n"
            "```\n토큰 100개, lanes 32 → ceil(100/32) = 4 사이클\n```\n\n"
            "★ **BRAM 워드와 헷갈리지 말 것.** 둘 다 '행의 가로 덩어리'이고 하필 실제 설정에서 "
            "둘 다 32라 구분이 안 간다. **레인은 연산 쪽, 워드는 메모리 쪽**이다."
        ),
        "confuse": ["word", "cycle"],
    },
    {
        "id": "word", "term": "BRAM 워드", "en": "word, word_tokens", "group": "하드웨어",
        "one": "메모리가 한 번에 읽는 단위. **역시 한 행의 가로 덩어리.**",
        "canvas": "grid", "highlight": {"kind": "chunk", "row": 2, "start": 0, "unit": "word_tokens", "style": "word"},
        "real": "32 토큰 / 워드",
        "detail": (
            "★ **BRAM 은 토큰을 하나씩 읽지 않는다.** 워드 안에 살아있는 토큰이 하나라도 있으면 "
            "**워드 전체를 읽어야 한다.**\n\n"
            "그래서 종단된 토큰이 흩어져 있으면 읽기가 거의 안 줄어든다. 실측에서 "
            "살아있는 토큰이 256 → 20 (92% 감소)인데 워드는 8 → 8 그대로였다.\n\n"
            "이 격차가 work-compaction 이 필요한 진짜 이유다."
        ),
        "confuse": ["lane", "token", "read_live"],
    },
    {
        "id": "cycle", "term": "사이클", "en": "cycle", "group": "하드웨어",
        "one": "한 레인 덩어리를 처리하는 시간. **평면 하나 = 여러 사이클.**",
        "canvas": "grid", "highlight": {"kind": "chunk", "row": 2, "start": 0, "unit": "lanes", "style": "cycle"},
        "real": "dense = n_planes × ceil(T/lanes)",
        "detail": (
            "★ **평면과 사이클은 다르다.** 평면은 데이터의 층이고 사이클은 시간이다. "
            "`lanes` 가 토큰 수보다 작으면 한 평면에 여러 사이클이 든다.\n\n"
            "그리고 사이클이 시간의 전부도 아니다 — BRAM 읽기가 `ceil(words / n_ports)` "
            "사이클을 따로 먹는다."
        ),
        "confuse": ["plane", "lane"],
    },
    {
        "id": "scattered", "term": "흩어짐 / 압축", "en": "scattered / compacted", "group": "하드웨어",
        "one": "생존 토큰이 **어디 있느냐.** 개수가 같아도 읽는 워드가 달라진다.",
        "canvas": "grid", "highlight": {"kind": "live"},
        "real": "word_reads_scattered / word_reads_compacted",
        "detail": (
            "★ **개수는 그대로인데 위치만 바꿔도 결과가 갈린다.** 실측 (128토큰, 이론 절감 70.7%):\n\n"
            "```\n            batch 워드 절감   compaction 워드 절감\n흩어져 죽음        0.0%              56.2%\n"
            "뭉쳐서 죽음       56.2%              56.2%\n```\n\n"
            "`compaction` 만이 배치와 무관하게 절감을 실현한다."
        ),
        "confuse": ["word"],
    },
    {
        "id": "port", "term": "BRAM 포트", "en": "n_ports", "group": "하드웨어",
        "one": "한 사이클에 읽을 수 있는 워드 수. `메모리 사이클 = ceil(워드 / 포트)`",
        "canvas": "grid", "highlight": {"kind": "chunk", "row": 2, "start": 0, "unit": "word_tokens", "style": "word"},
        "real": "2",
        "detail": (
            "오랫동안 **설정에만 있고 어떤 계산에도 읽히지 않던 값**이었다. "
            "그래서 '워드를 몇 번 읽는가'는 세면서 '그게 몇 사이클인가'는 세지 않았다.\n\n"
            "현재 설정이 안전한 것은 우연이다 — `word_tokens(32) == lanes(32)` 라 "
            "워드 하나가 정확히 한 연산 사이클을 채운다. 워드를 좁히면 즉시 메모리 병목이 된다."
        ),
        "confuse": ["word"],
    },
]

#: ★ 헷갈리기 쉬운 짝 — 같은 격자에 나란히 올려 차이를 보인다.
_PAIRS: list[dict[str, Any]] = [
    {
        "a": "lane", "b": "word",
        "title": "레인 vs BRAM 워드",
        "why": (
            "**둘 다 한 행의 가로 덩어리**이고, 실제 설정에서 **둘 다 32**라 그림으로는 구분이 안 간다. "
            "하지만 레인은 **연산**이 한 번에 처리하는 폭이고 워드는 **메모리**가 한 번에 읽는 폭이다. "
            "`word_tokens` 를 8로 줄이면 워드 하나가 레인을 못 채워 즉시 메모리 병목이 된다."
        ),
    },
    {
        "a": "plane", "b": "cycle",
        "title": "비트평면 vs 사이클",
        "why": (
            "평면은 **데이터의 층**, 사이클은 **시간**이다. 토큰이 레인보다 많으면 "
            "평면 하나에 여러 사이클이 든다. `dense = n_planes × ceil(T/lanes)` 에서 "
            "두 개념이 곱해지는 것이 보인다."
        ),
    },
    {
        "a": "read_live", "b": "word",
        "title": "읽기 마스크 vs 워드 수",
        "why": (
            "★ **브리지의 단위 문제.** `read_live` 는 **토큰 마스크**(bool 격자), "
            "`account_step()` 출력은 **BRAM 워드 수**(정수)다. "
            "섞이면 모든 절감 수치가 무의미해진다 — 두 팀이 합동으로 확인해야 하는 지점이다."
        ),
    },
    {
        "a": "r_m", "b": "theta",
        "title": "상한 vs 문턱",
        "why": (
            "`S_m + R_m` 은 **이 토큰이 최대 얼마까지 갈 수 있는가**, "
            "θ 는 **살아남으려면 얼마를 넘어야 하는가**다. "
            "상한이 θ 아래로 떨어지는 순간 종단이 확정된다 — 두 선이 만나는 자리가 판정이다."
        ),
    },
]


def build() -> dict[str, Any]:
    """화면이 그릴 사전 전체."""
    real = _real_spec()
    groups: list[str] = []
    for t in _TERMS:
        if t["group"] not in groups:
            groups.append(t["group"])
    return {
        "demo": DEMO,
        "real": real,
        "groups": groups,
        "terms": _TERMS,
        "pairs": _PAIRS,
        "scale_note": (
            f"그림은 축소판이다 (토큰 {DEMO['n_tokens']} · 레인 {DEMO['lanes']} · "
            f"워드 {DEMO['word_tokens']}). 실제는 레인 {real['lanes']} · "
            f"워드 {real['word_tokens']} 이며, **레인 == 워드 관계는 축소판에서도 유지**했다."
        ),
    }
