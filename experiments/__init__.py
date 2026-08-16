"""실험 스크립트 모음.

각 모듈은 run(cfg, ...) -> list[dict] 를 제공하고,
결과를 outputs/raw/<name>.csv 에 저장한다.
그림 생성은 outputs/raw/ 만 읽으므로, 실험을 다시 돌리지 않고도
figure 를 재생성할 수 있다.
"""

from __future__ import annotations

from src.config import load_config
from src.dataset import snapshot_from_config
from src.decode_loop import DecodeWorkbench, workbench_from_config
from src.schedule import bram_from_config, spec_from_config

__all__ = [
    "load_config",
    "snapshot_from_config",
    "workbench_from_config",
    "spec_from_config",
    "bram_from_config",
    "build_workbench",
]


def build_workbench(cfg, seq_len: int | None = None, seed: int = 0) -> DecodeWorkbench:
    """설정에서 스냅샷 -> 워크벤치까지 한 번에.

    부분 내적은 여기서 한 번만 계산되고 이후 모든 스윕이 재사용한다.
    """
    snap = snapshot_from_config(cfg, seed=seed, seq_len=seq_len)
    return workbench_from_config(cfg, snap, seq_len=seq_len) if seq_len else workbench_from_config(cfg, snap)
