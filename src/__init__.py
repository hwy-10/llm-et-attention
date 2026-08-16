"""LLM 어텐션 가속기 — 상위 비트 우선 계산 + 조기 종단, 소프트웨어 검증 코어.

의존성: numpy 만. (torch / matplotlib / pandas / PyYAML 은 전부 선택)

빠른 사용법
-----------
    from src.config import load_config
    from src.dataset import snapshot_from_config
    from src.decode_loop import workbench_from_config, run_decode

    cfg  = load_config()
    snap = snapshot_from_config(cfg)
    wb   = workbench_from_config(cfg, snap)     # 부분 내적 1회 전처리

    for design in ("baseline", "seq", "exact", "approx"):
        r = run_decode(wb, design=design, top_k=8, margin=0.1)
        print(design, r.summary["mean_term_plane"], r.summary["read_saving_bram"])
"""

from .config import Config, load_config
from .designs import DESIGNS, design_label

__all__ = ["Config", "load_config", "DESIGNS", "design_label"]
__version__ = "0.1.0"
