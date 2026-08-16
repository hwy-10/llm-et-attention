"""논문용 그래프 생성.

matplotlib 이 없으면 임포트는 되고 호출 시점에 안내만 띄운다
(코어 실험은 matplotlib 없이도 전부 돌아간다).

색 규약
-------
검증된 3슬롯 팔레트를 쓴다. 대비/색각 이상 분리도 검증 완료:
  라이트 all-pairs  CVD ΔE 9.2 / 일반시야 ΔE 24.0
  다크  all-pairs  CVD ΔE 9.4 / 일반시야 ΔE 20.9

  ② seq    = aqua      (순차 전환 비용)
  ③ exact  = blue      (핵심 결과)
  ④ approx = orange    (절감-정확도 곡선)
  ① baseline = muted 회색 파선 — 계열이 아니라 **기준선**이므로 색을 쓰지 않는다

라이트 모드에서 aqua 는 배경 대비 3:1 미만이므로 **직접 라벨이 필수**다.
이 모듈의 모든 선 그래프는 직접 라벨을 붙인다 (범례도 함께).

축 규약
-------
이중 축(twin axis)을 절대 쓰지 않는다. 가이드 그림 8.1 의 "절감량 곡선"과
"정확도 손실 곡선"은 스케일이 다르므로 **두 개의 패널**로 나란히 그린다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

_MISSING = """\
[visualization] matplotlib 이 설치되어 있지 않습니다.

  pip install "matplotlib>=3.7"

그래프 없이도 실험은 전부 실행되며 결과는 outputs/raw/ 에 CSV 로 남습니다.
설치 후 `python run_paper_experiments.py --figures-only` 로 그림만 다시 뽑으면 됩니다.
"""

# --- 팔레트 ----------------------------------------------------------------
LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "s1": "#2a78d6",   # blue
    "s2": "#eb6834",   # orange
    "s3": "#1baf7a",   # aqua
}
DARK = {
    "surface": "#1a1a19",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "s1": "#3987e5",
    "s2": "#d95926",
    "s3": "#199e70",
}

# 설계 -> 색 슬롯 (엔티티에 고정. 계열 개수가 바뀌어도 색이 안 바뀐다)
DESIGN_SLOT = {"seq": "s3", "exact": "s1", "approx": "s2", "baseline": "muted"}
DESIGN_LABEL = {
    "baseline": "① 병렬 INT8 MAC",
    "seq": "② 비트평면 순차",
    "exact": "③ 정확 종단",
    "approx": "④ 근사 종단",
}


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_MISSING) from exc


def palette(theme: str = "light") -> dict:
    return DARK if theme == "dark" else LIGHT


# 축 라벨이 한글이므로 한글 글리프가 있는 폰트를 찾아 쓴다.
# 없으면 DejaVu Sans 로 떨어지고 한글이 네모로 나온다 -> 경고를 띄운다.
_KOREAN_FONTS = (
    "Malgun Gothic",        # Windows 기본
    "AppleGothic",          # macOS
    "NanumGothic",
    "Noto Sans KR",
    "Noto Sans CJK KR",
    "NanumBarunGothic",
    "Gulim",
)
_font_warned = False


def resolve_korean_font() -> str | None:
    """설치된 한글 폰트 이름을 반환. 없으면 None."""
    try:
        from matplotlib import font_manager
    except ImportError:
        return None
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in _KOREAN_FONTS if f in available), None)


def apply_style(theme: str = "light", font_family: str | None = None, base_size: float = 9.0):
    """논문용 스타일.

    font_family=None 이면 한글 폰트를 자동 탐지한다.
    IEEE 카메라레디에 맞추려면 'Times New Roman' 등을 직접 넘긴다
    (단, 한글 라벨은 깨지므로 라벨을 영문으로 바꿔야 한다).
    """
    global _font_warned
    plt = _mpl()
    c = palette(theme)

    if font_family is None:
        font_family = resolve_korean_font() or "DejaVu Sans"
        if font_family == "DejaVu Sans" and not _font_warned:
            _font_warned = True
            print("  [viz] 한글 폰트를 찾지 못했습니다. 축 라벨이 네모로 나올 수 있습니다.\n"
                  "        Windows: Malgun Gothic, Linux: fonts-nanum 설치를 권장합니다.")

    plt.rcParams.update({
        "figure.facecolor": c["surface"],
        "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"],
        "font.family": font_family,
        "axes.unicode_minus": False,   # 한글 폰트에는 U+2212 가 없는 경우가 많다
        "font.size": base_size,
        "axes.labelsize": base_size,
        "axes.titlesize": base_size + 1,
        "legend.fontsize": base_size - 1,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "axes.edgecolor": c["axis"],
        "axes.labelcolor": c["ink"],
        "text.color": c["ink"],
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "axes.grid": True,
        "grid.color": c["grid"],
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,   # 벡터 폰트 임베딩 (편집 가능)
        "ps.fonttype": 42,
    })
    return plt, c


def _direct_label(ax, x, y, text, color, dx=4, dy=0.0):
    """마지막 점 옆에 직접 라벨. 라이트 모드 저대비 색의 relief 규칙 충족.

    dy 로 세로 오프셋을 줘서 계열이 겹칠 때 라벨 충돌을 피한다.
    """
    if len(x) == 0:
        return
    ax.annotate(
        text, xy=(x[-1], y[-1]), xytext=(dx, dy), textcoords="offset points",
        color=color, va="center", ha="left", fontsize=7.5, clip_on=False,
    )


def _place_labels(ax, entries, min_sep_frac: float = 0.055):
    """겹치지 않게 직접 라벨을 배치한다.

    entries: [(x_last, y_last, text, color), ...]
    y 가 가까운 라벨을 아래로 밀어 최소 간격을 확보한다.
    모든 plot 이 끝난 뒤 (ylim 이 확정된 뒤) 호출할 것.
    """
    ymin, ymax = ax.get_ylim()
    sep = min_sep_frac * (ymax - ymin)
    placed: list[float] = []
    for x, y, text, color in sorted(entries, key=lambda e: -e[1]):
        if placed and (placed[-1] - y) < sep:
            y = placed[-1] - sep
        placed.append(y)
        ax.annotate(text, xy=(x, y), xytext=(5, 0), textcoords="offset points",
                    color=color, va="center", ha="left", fontsize=7.5, clip_on=False)


def _rolling_mean(y, window: int):
    """이동 평균. (x, y) 를 함께 잘라 반환한다. pandas 불필요."""
    import numpy as np

    y = np.asarray(y, dtype=float)
    w = int(max(1, min(window, y.size)))
    if w <= 1:
        return np.arange(y.size), y
    kernel = np.ones(w) / w
    sm = np.convolve(y, kernel, mode="valid")
    idx = np.arange(w - 1, y.size)
    return idx, sm


# 계열 구분을 색에만 맡기지 않는다 (색각 이상 · 흑백 인쇄 대비)
MARKERS = ("o", "s", "^", "D")


def save(fig, name: str, subdir: str | Path | None = None) -> list[Path]:
    """PDF(벡터) + PNG 동시 저장."""
    from .io import FIG_DIR, ensure_dirs

    ensure_dirs()
    base = Path(subdir) if subdir else FIG_DIR
    base.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("pdf", "png"):
        p = base / f"{name}.{ext}"
        fig.savefig(p)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# 그림 8.1 — ★ 최종 산출물 ★
# ---------------------------------------------------------------------------
def fig_tradeoff(
    records: Sequence[dict],
    *,
    saving_key: str = "read_saving_bram",
    saving_key_ideal: str = "read_saving_ideal",
    accuracy_key: str = "top8_retention",
    name: str = "fig8_1_tradeoff",
    theme: str = "light",
    title: str = "",
):
    """여유값(margin) 대비 절감량 / 정확도 손실 — 두 패널.

    ★ 이중 축을 쓰지 않는다 ★ 스케일이 다른 두 지표는 패널을 나눈다.
    절감량 패널에는 '이론'과 '실현'을 함께 그려 격차를 드러낸다.
    """
    plt, c = apply_style(theme)
    recs = sorted(records, key=lambda r: r.get("margin", 0.0))
    x = [r["margin"] for r in recs]
    y_real = [r.get(saving_key, 0.0) * 100 for r in recs]
    y_ideal = [r.get(saving_key_ideal, 0.0) * 100 for r in recs]
    y_acc = [(1.0 - r.get(accuracy_key, 1.0)) * 100 for r in recs]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(x, y_ideal, color=c["muted"], linestyle="--", marker="o", label="이론 (토큰 단위)")
    ax.plot(x, y_real, color=c["s1"], marker="o", label="실현 (BRAM 워드 단위)")
    _direct_label(ax, x, y_ideal, "이론", c["muted"])
    _direct_label(ax, x, y_real, "실현", c["s1"])
    ax.set_xlabel("θ 여유값 (margin)")
    ax.set_ylabel("메모리 읽기 절감 (%)")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(x, y_acc, color=c["s2"], marker="o")
    _direct_label(ax, x, y_acc, f"top-k 손실", c["s2"])
    ax.set_xlabel("θ 여유값 (margin)")
    ax.set_ylabel("상위 k 보존 손실 (%p)")
    ax.axhline(0.0, color=c["axis"], linewidth=0.8)

    if title:
        fig.suptitle(title, fontsize=10)
    return fig, save(fig, name)


# ---------------------------------------------------------------------------
def fig_termination_profile(
    traces: dict[str, dict],
    *,
    name: str = "fig_termination_profile",
    theme: str = "light",
    n_planes: int = 8,
    smooth: int = 24,
):
    """디코드 스텝(= 문맥 길이)에 따른 평균 종단 시점.

    traces: {설계이름: {"n_active": [...], "mean_term_plane": [...]}, ...}

    스텝별 값은 잡음이 심하므로 이동 평균을 굵게, 원값을 옅게 겹쳐 그린다
    (원값을 숨기지 않으면서 추세를 읽을 수 있게).
    종단 개념이 없는 ①/② 는 이 그림에서 제외하고 상단 기준선으로만 표시한다.
    """
    import numpy as np

    plt, c = apply_style(theme)
    fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)

    shown = {d: tr for d, tr in traces.items() if d not in ("baseline", "seq")}
    labels: list[tuple] = []
    for design, tr in sorted(shown.items()):
        col = c[DESIGN_SLOT.get(design, "s1")]
        x = np.asarray(tr["n_active"], dtype=float)
        y = np.asarray(tr["mean_term_plane"], dtype=float)
        ax.plot(x, y, color=col, linewidth=0.6, alpha=0.28)          # 원값 (옅게)
        idx, sm = _rolling_mean(y, smooth)
        ax.plot(x[idx], sm, color=col, linewidth=1.8,
                label=DESIGN_LABEL.get(design, design))               # 추세 (굵게)
        labels.append((x[idx][-1], sm[-1], DESIGN_LABEL.get(design, design).split()[0], col))

    ax.axhline(n_planes, color=c["axis"], linewidth=0.9, linestyle="--")
    ax.annotate("종단 없음 (①②)", xy=(0.02, n_planes), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=7, color=c["muted"])
    ax.set_xlabel("문맥 길이 (활성 토큰 수)")
    ax.set_ylabel("평균 종단 평면")
    ax.set_ylim(0, n_planes + 0.9)
    ax.set_xlim(right=float(max(np.max(t["n_active"]) for t in shown.values())) * 1.16)
    if len(shown) >= 2:
        ax.legend(loc="lower left")
    _place_labels(ax, labels)
    return fig, save(fig, name)


# ---------------------------------------------------------------------------
def fig_read_realization(
    records: Sequence[dict],
    *,
    name: str = "fig_read_realization",
    theme: str = "light",
):
    """★ BRAM 워드폭이 읽기 절감 실현을 어떻게 깎는가 ★

    x = word_tokens, 계열 = 스케줄 정책. 이론값은 회색 파선 기준선.
    """
    plt, c = apply_style(theme)
    fig, ax = plt.subplots(figsize=(4.0, 2.7), constrained_layout=True)

    policies = []
    for r in records:
        p = r.get("schedule_policy") or "none"
        if p not in policies:
            policies.append(p)
    slots = ["s1", "s2", "s3"]
    labels: list[tuple] = []

    def _dedup(rows):
        """워드폭마다 대표값 하나 (같은 정책의 변형은 최댓값으로)."""
        best: dict[int, dict] = {}
        for r in rows:
            wt = r.get("word_tokens", 1)
            if wt not in best or r.get("read_saving_bram", 0) > best[wt].get("read_saving_bram", 0):
                best[wt] = r
        return [best[k] for k in sorted(best)]

    ideal = _dedup([r for r in records if r.get("schedule_policy") == policies[0]])
    if ideal:
        xi = [r["word_tokens"] for r in ideal]
        yi = [r["read_saving_ideal"] * 100 for r in ideal]
        ax.plot(xi, yi, color=c["muted"], linestyle="--", marker="o",
                markersize=4, label="이론 상한")
        labels.append((xi[-1], yi[-1], "이론", c["muted"]))

    for i, pol in enumerate(policies):
        sub = _dedup([r for r in records if r.get("schedule_policy") == pol])
        if not sub:
            continue
        col = c[slots[i % len(slots)]]
        x = [r["word_tokens"] for r in sub]
        y = [r["read_saving_bram"] * 100 for r in sub]
        # 마커를 계열마다 다르게 — 곡선이 겹쳐도 구분된다 (secondary encoding)
        ax.plot(x, y, color=col, marker=MARKERS[i % len(MARKERS)],
                markersize=5, markerfacecolor=c["surface"], markeredgewidth=1.4,
                label=pol)
        labels.append((x[-1], y[-1], pol, col))

    ax.set_xscale("log", base=2)
    ax.set_xlim(right=max(r.get("word_tokens", 1) for r in records) * 3.2)
    ax.set_xlabel("BRAM 워드당 토큰 수")
    ax.set_ylabel("메모리 읽기 절감 (%)")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.42))
    _place_labels(ax, labels)
    return fig, save(fig, name)


# ---------------------------------------------------------------------------
def fig_breakeven(
    grid,
    cycle_ratios: Sequence[float],
    fmax_ratios: Sequence[float],
    *,
    name: str = "fig_breakeven",
    theme: str = "light",
):
    """실효 speedup 등고선. 1.0 등고선이 손익분기선이다 (가이드 6.3-(4))."""
    import numpy as np

    plt, c = apply_style(theme)
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)

    g = np.asarray(grid)
    # 발산형이 아니라 크기 스케일 -> 단일 색상 순차 램프 (blue 100→700)
    seq_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]
    )
    im = ax.pcolormesh(fmax_ratios, cycle_ratios, g, cmap=seq_cmap, shading="auto")
    cs = ax.contour(fmax_ratios, cycle_ratios, g, levels=[1.0], colors=[c["ink"]], linewidths=1.4)
    ax.clabel(cs, fmt={1.0: "손익분기"}, fontsize=7)
    fig.colorbar(im, ax=ax, label="실효 speedup")
    ax.set_xlabel("Fmax 비 (제안 / 기준)")
    ax.set_ylabel("사이클 비 (제안 / 기준)")
    ax.grid(False)
    return fig, save(fig, name)


# ---------------------------------------------------------------------------
def fig_design_comparison(
    records: Sequence[dict],
    *,
    metric: str = "total_cycles",
    ylabel: str = "총 사이클",
    name: str = "fig_design_comparison",
    theme: str = "light",
):
    """네 설계 비교 막대. 인접 막대 사이에 2px 표면 간격을 둔다."""
    plt, c = apply_style(theme)
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)

    order = [d for d in ("baseline", "seq", "exact", "approx")
             if any(r.get("design") == d for r in records)]
    vals, cols = [], []
    for d in order:
        sub = [r for r in records if r.get("design") == d]
        vals.append(sub[0].get(metric, 0))
        cols.append(c[DESIGN_SLOT.get(d, "s1")])

    xs = range(len(order))
    ax.bar(xs, vals, color=cols, width=0.62, linewidth=1.0, edgecolor=c["surface"])
    for i, v in enumerate(vals):                    # 선택적 직접 라벨 (막대마다 값)
        ax.annotate(f"{v:,.0f}", xy=(i, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color=c["ink2"])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([DESIGN_LABEL.get(d, d) for d in order], rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    return fig, save(fig, name)
