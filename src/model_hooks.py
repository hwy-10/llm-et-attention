"""실제 Llama 3.2 1B 에서 q / K 텐서를 캡처해 캐시에 덤프한다.

★ 왜 캐시가 필수인가 ★
exp2 만 해도 margin(9) x top_k(3) x seed(3) = 81 회 디코드 루프다.
매번 모델을 forward 하면 스윕이 불가능하다. **한 번 덤프하고 numpy 로만 돈다.**

torch / transformers 가 없으면 임포트 시점에 실패하지 않고, capture_qk() 호출
시점에 안내를 띄운다. 그 경우 src/dataset.py 의 합성 텐서로 전체가 그대로 돈다.

사용법
------
    python -m src.model_hooks --seq-len 512 --layer 8 --head 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import load_config
from .dataset import QKSnapshot, cached_snapshot_path

_MISSING = """\
[model_hooks] torch / transformers 가 설치되어 있지 않습니다.

  pip install "torch>=2.0" "transformers>=4.40" "datasets>=2.18"

설치 전에도 src/dataset.py 의 합성 텐서로 모든 실험이 실행됩니다.
다만 논문 본문 수치는 실제 캡처 텐서로 다시 뽑아야 합니다.
"""


def _require_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise RuntimeError(_MISSING) from exc
    import torch
    import transformers

    return torch, transformers


def capture_qk(
    hf_id: str,
    text: str,
    *,
    layer_idx: int = 8,
    head_idx: int = 0,
    seq_len: int = 512,
    device: str = "cpu",
) -> QKSnapshot:
    """지정 층/헤드의 q, k 를 토큰 인덱스로 정렬해 캡처한다.

    한계로 명시할 것: RoPE 적용 이전의 q_proj / k_proj 출력을 잡는다.
    RoPE 는 회전이므로 내적 분포의 스케일을 크게 바꾸지 않으나,
    토큰 간 상대 위치 효과는 반영되지 않는다.
    """
    torch, _ = _require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device)
    model.eval()

    attn = model.model.layers[layer_idx].self_attn
    head_dim = getattr(
        attn, "head_dim", model.config.hidden_size // model.config.num_attention_heads
    )
    n_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)

    grabbed: dict = {}
    h1 = attn.q_proj.register_forward_hook(lambda m, i, o: grabbed.__setitem__("q", o.detach()))
    h2 = attn.k_proj.register_forward_hook(lambda m, i, o: grabbed.__setitem__("k", o.detach()))
    try:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
        with torch.no_grad():
            model(**ids)
    finally:
        h1.remove()
        h2.remove()

    q = grabbed["q"][0].reshape(-1, grabbed["q"].shape[-1] // head_dim, head_dim)
    k = grabbed["k"][0].reshape(-1, n_kv_heads, head_dim)
    kv_head = head_idx % n_kv_heads

    q_np = q[:, head_idx, :].to(torch.float32).cpu().numpy()
    k_np = k[:, kv_head, :].to(torch.float32).cpu().numpy()
    n = min(len(q_np), len(k_np), seq_len)

    return QKSnapshot(
        q=np.ascontiguousarray(q_np[:n], dtype=np.float64),
        k=np.ascontiguousarray(k_np[:n], dtype=np.float64),
        source=f"{hf_id}:L{layer_idx}:H{head_idx}",
        meta={"rope_applied": False, "n_kv_heads": int(n_kv_heads)},
    )


_DEFAULT_TEXT = (
    "In autoregressive decoding the key-value cache is written once per token and read "
    "on every subsequent step, which makes the memory read path the bottleneck rather "
    "than the arithmetic units. Attention scores are highly skewed after the softmax, so "
    "only a small number of tokens carry meaningful weight in the final output. "
) * 24


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Capture real q/K tensors into cache/tensors/")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--head", type=int, default=None)
    ap.add_argument("--text-file", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args(argv)

    cfg = load_config()
    dec = cfg.get("model.decode", {}) or {}
    seq_len = args.seq_len or int(dec.get("seq_len", 512))
    layer_idx = args.layer if args.layer is not None else int(dec.get("layer_idx", 8))
    head_idx = args.head if args.head is not None else int(dec.get("head_idx", 0))
    hf_id = cfg.get("model.model.hf_id", "meta-llama/Llama-3.2-1B")
    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else _DEFAULT_TEXT

    try:
        snap = capture_qk(
            hf_id, text, layer_idx=layer_idx, head_idx=head_idx,
            seq_len=seq_len, device=args.device,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    out = cached_snapshot_path(cfg, seq_len)
    snap.save(out)
    print(f"[model_hooks] saved {snap.n_tokens} tokens (head_dim={snap.head_dim}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
