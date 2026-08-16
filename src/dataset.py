"""평가용 Q / K 텐서 공급.

두 경로가 있고 인터페이스는 같다.

  1) 실제 모델 캡처 — src/model_hooks.py (torch + transformers 필요)
  2) 합성 텐서      — 이 파일. torch 없이도 전체 파이프라인이 돈다.

디코드 루프를 돌기 위해 q 와 k 는 **같은 길이(seq_len)** 로 정렬되어 있어야 한다.
  step s 는 q[s] 로 k[0:s] 를 본다.

합성 텐서는 그냥 가우시안이 아니다. 가이드 4.1절의 "소수 토큰이 softmax 를
지배한다"를 attention sink 로 재현한다. 이 성질이 없으면 종단이 거의 일어나지
않아 실험 자체가 무의미해진다.

주의: 합성 데이터 수치는 **경향 확인용**이다. 논문 본문 수치는
      model_hooks.py 로 캡처한 실제 텐서로 다시 뽑아야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .seeding import data_rng


@dataclass
class QKSnapshot:
    """토큰 인덱스로 정렬된 q, k 쌍."""

    q: np.ndarray            # (seq_len, head_dim)
    k: np.ndarray            # (seq_len, head_dim)
    source: str = "synthetic"
    meta: dict | None = None

    @property
    def head_dim(self) -> int:
        return self.k.shape[1]

    @property
    def n_tokens(self) -> int:
        return self.k.shape[0]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q=self.q, k=self.k, source=np.array(self.source))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QKSnapshot":
        d = np.load(Path(path), allow_pickle=False)
        return cls(q=d["q"], k=d["k"], source=str(d["source"]))


def synthetic_qk(
    seq_len: int = 512,
    head_dim: int = 64,
    *,
    q_std: float = 1.0,
    k_std: float = 1.0,
    sink_frac: float = 0.04,
    sink_gain: float = 3.5,
    target_logit_std: float = 5.0,
    seed: int = 0,
) -> QKSnapshot:
    """attention sink 를 포함한 합성 Q/K.

    target_logit_std
        ★ 그냥 표준정규 q,k 를 쓰면 로짓 표준편차가 1 근처가 되어 softmax 가
          거의 균등해진다. 실제 LLM 어텐션은 훨씬 뾰족하다(로짓 std 대략 3~8).
          평평한 분포에서는 종단도 안 일어나고 top-k 근사 손실도 과대평가된다.
          그래서 로짓 스케일을 목표값에 맞춰 보정한다.
          (양자화 후 정수 점수는 스케일 불변이므로 종단 거동 자체는 이 값과
           무관하고, softmax 집중도만 현실적으로 맞춰진다.)
    """
    rng = data_rng(seed)
    q = rng.normal(0.0, q_std, size=(seq_len, head_dim))
    k = rng.normal(0.0, k_std, size=(seq_len, head_dim))

    n_sink = max(1, int(round(sink_frac * seq_len)))
    # 토큰 0 은 항상 sink (실제 LLM 관측과 일치)
    sink_idx = np.unique(
        np.concatenate([[0], rng.choice(seq_len, size=n_sink, replace=False)])
    )

    q_dir = q.mean(axis=0)
    nrm = np.linalg.norm(q_dir)
    q_dir = q_dir / nrm if nrm > 0 else np.ones(head_dim) / np.sqrt(head_dim)

    for j in sink_idx:
        mixed = 0.75 * q_dir * np.linalg.norm(k[j]) + 0.25 * k[j]
        k[j] = sink_gain * mixed

    if target_logit_std and target_logit_std > 0:
        probe = min(seq_len, 128)
        logits = (q[:probe] @ k[:probe].T) / np.sqrt(head_dim)
        cur = float(np.std(logits))
        if cur > 0:
            q *= target_logit_std / cur

    return QKSnapshot(
        q=q, k=k, source="synthetic",
        meta={
            "sink_idx": sink_idx.tolist(),
            "sink_gain": sink_gain,
            "target_logit_std": target_logit_std,
            "seed": seed,
        },
    )


def cached_snapshot_path(cfg, seq_len: int) -> Path:
    """model_hooks.py 가 덤프한 실제 텐서의 표준 경로."""
    from .config import PROJECT_ROOT

    dec = cfg.get("model.decode", {}) or {}
    name = cfg.get("model.model.name", "model")
    return (
        PROJECT_ROOT / "cache" / "tensors"
        / f"{name}_L{dec.get('layer_idx', 0)}_H{dec.get('head_idx', 0)}_T{seq_len}.npz"
    )


def snapshot_from_config(cfg, seed: int = 0, seq_len: int | None = None) -> QKSnapshot:
    """캐시된 실제 텐서가 있으면 그걸 쓰고, 없으면 합성 텐서."""
    dec = cfg.get("model.decode", {}) or {}
    syn = cfg.get("model.synthetic", {}) or {}
    seq_len = int(seq_len or dec.get("seq_len", 512))
    head_dim = int(cfg.head_dim)

    cache = cached_snapshot_path(cfg, seq_len)
    if cache.exists():
        snap = QKSnapshot.load(cache)
        if snap.n_tokens >= seq_len:
            return QKSnapshot(
                q=snap.q[:seq_len], k=snap.k[:seq_len],
                source=snap.source, meta={"cache": str(cache)},
            )
    # 캐시가 요청 길이보다 짧아도 다른 길이용 캐시가 있을 수 있다
    for other in sorted((cache.parent).glob("*.npz")) if cache.parent.exists() else []:
        snap = QKSnapshot.load(other)
        if snap.n_tokens >= seq_len and snap.head_dim == head_dim:
            return QKSnapshot(
                q=snap.q[:seq_len], k=snap.k[:seq_len],
                source=snap.source, meta={"cache": str(other)},
            )

    return synthetic_qk(
        seq_len=seq_len, head_dim=head_dim,
        q_std=float(syn.get("q_std", 1.0)),
        k_std=float(syn.get("k_std", 1.0)),
        sink_frac=float(syn.get("sink_frac", 0.04)),
        sink_gain=float(syn.get("sink_gain", 3.5)),
        target_logit_std=float(syn.get("target_logit_std", 5.0)),
        seed=seed,
    )
