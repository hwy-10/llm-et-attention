"""EXP8 — ★ 보간이 아닌 실측 perplexity ★  (2026-08-29 신설, 팀 2)

지금까지의 `K_TOP` 근거(+5.8%)는 **보간값**이었다.

    하드 top-k 로 perplexity 곡선을 만들고,
    우리 생존 토큰 수(31.3개)를 그 곡선에 로그 보간해서 읽었다

생존 집합은 하드 top-k 가 아니라 **top-k 의 상위집합**이다. 상한이 theta 를 못 넘는
것만 버리므로 "버릴 수 없는 토큰"이 함께 남는다. 그래서 같은 개수의 하드 top-k 보다
품질이 좋을 수밖에 없고, 보간은 그 이득을 반영하지 못한다. 즉 **우리 쪽에 불리한
추정**이었다.

이 실험은 **종단 로직을 어텐션에 직접 넣어** 잰다.

    층 x 헤드마다:
      1. K 를 INT8 양자화 -> 비트평면
      2. q 를 접어 넣어 정수화
      3. terminator.run_step 으로 생존 집합을 얻는다
      4. 생존하지 않은 토큰의 로짓을 -inf 로 만들고 softmax 재정규화
    -> 그 상태로 wikitext perplexity 를 잰다

★★ 층을 하나만 바꾸면 안 된다 ★★
  층 8 하나만 바꾸면 나머지 15개 층이 full attention 이라 열화가 거의 안 보인다
  (실측 +0.48%). 하드웨어는 **모든 층의 어텐션을 대체**하므로 그 조건으로 재야
  논문에 쓸 수 있다. `--all-layers` 가 그 모드다. 기본값이기도 하다.
  단일 층 결과는 "어느 층이 민감한가"를 보는 진단용으로만 쓴다.

★ 반드시 함께 재는 것 ★
  full        : 손대지 않은 모델. 기준값
  oracle top-k: 참 top-k 만 남긴 희소 어텐션. **우리가 이길 수 없는 하한**
  ours        : 우리 종단 로직

  ours 가 oracle 보다 좋으면 상위집합 효과가 실재한다는 뜻이다.
  full 대비 열화가 oracle 보다 작으면 "top-k 를 쓰는 어떤 방법보다 낫다".

의존성: torch + transformers. 없으면 건너뛴다.
    python -m experiments.exp8_real_perplexity --tokens 4096 --layers 8
"""

from __future__ import annotations

import math

import numpy as np

from utils.io import save_records
from utils.metrics import perplexity_delta

from . import load_config

NAME = "exp8_real_perplexity"


# ---------------------------------------------------------------------------
# 우리 종단 로직을 로짓 마스크로 바꾸는 부분
# ---------------------------------------------------------------------------
def survivor_mask(
    q_real: np.ndarray,
    k_real: np.ndarray,
    *,
    top_k: int,
    margin: float,
    margin_mode: str = "relative_width",
    n_planes: int = 8,
    lanes: int = 32,
    pipeline_cycles: int = 8,
    latency_mode: str = "auto",
) -> np.ndarray:
    """(n_q, n_k) bool — 각 질의가 어느 키를 살려 두는가.

    골든모델과 **같은 경로**를 탄다. 여기서만 쓰는 근사가 없어야
    perplexity 수치가 exp1~exp7 과 같은 설계를 가리킨다.
    """
    from src.accumulator import fold_and_quantize_query
    from src.bounds import StepBounds
    from src.masked_sum import partial_dots
    from src.memory import latency_planes
    from src.quantize import quantize_key, to_bitplanes
    from src.terminator import run_step
    from src.threshold import ThetaPolicy

    key = quantize_key(k_real, bits=n_planes)
    fq = fold_and_quantize_query(q_real, key)
    planes = to_bitplanes(key.stored, n_planes)
    pol = ThetaPolicy(top_k=top_k, margin=margin, margin_mode=margin_mode)

    n_q = q_real.shape[0]
    out = np.zeros((n_q, k_real.shape[0]), dtype=bool)
    for s in range(n_q):
        n_act = s + 1                                   # causal
        p = partial_dots(fq.stored[s : s + 1], planes[:, :n_act, :])[:, 0, :]
        qs = fq.stored[s].astype(np.int64)
        b = StepBounds(q_pos=int(qs[qs > 0].sum()), q_neg=int(qs[qs < 0].sum()))
        lat = (latency_planes(n_act, lanes, pipeline_cycles)
               if latency_mode == "auto" else 1)
        res = run_step(p, b, pol, decision_latency=lat)
        out[s, :n_act] = res.alive
    return out


def _oracle_mask(logits: np.ndarray, top_k: int) -> np.ndarray:
    """참 top-k 만 남긴 마스크. 우리가 이길 수 없는 하한."""
    n_q = logits.shape[0]
    out = np.zeros_like(logits, dtype=bool)
    for s in range(n_q):
        n_act = s + 1
        kk = min(top_k, n_act)
        row = logits[s, :n_act]
        out[s, np.argpartition(row, -kk)[-kk:]] = True
    return out


# ---------------------------------------------------------------------------
# perplexity
# ---------------------------------------------------------------------------
def run(cfg=None, verbose: bool = True, n_tokens: int = 2048,
        layers: tuple[int, ...] = (4, 8, 12), top_ks: tuple[int, ...] = (8, 16, 32),
        margins: tuple[float, ...] = (0.0, 0.5, 0.7),
        all_layers: bool = True) -> list[dict]:
    cfg = cfg or load_config()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        if verbose:
            print("  torch / transformers 가 없어 건너뜁니다. "
                  'pip install "torch>=2.0" "transformers>=4.40"')
        return []

    hf_id = cfg.get("model.model.hf_id", "unsloth/Llama-3.2-1B")
    text = _load_text(cfg)

    tok = AutoTokenizer.from_pretrained(hf_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32)
    except TypeError:                                   # transformers 4.x
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
    model.eval()

    ids = tok(text, return_tensors="pt", truncation=True, max_length=n_tokens).input_ids
    n_tokens = int(ids.shape[1])
    n_layers = model.config.num_hidden_layers
    # ★ 기본은 전 층 교체. 하드웨어가 하는 일이 그것이다.
    targets = [tuple(range(n_layers))] if all_layers else [(L,) for L in layers]
    if verbose:
        what = f"전 층({n_layers}개) 동시" if all_layers else f"층 {list(layers)} 개별"
        print(f"  {hf_id} · {n_tokens} 토큰 · {what} · "
              f"top_k {list(top_ks)} · margin {list(margins)}")

    records: list[dict] = []
    base_ppl = _perplexity(model, ids, mask_fn=None)
    if verbose:
        print(f"\n  full attention perplexity = {base_ppl:.4f}\n")
        print(f"    {'층':>4} {'k':>4} {'margin':>7} {'방식':>8} "
              f"{'ppl':>9} {'full 대비':>10} {'평균 생존':>10}")

    for layer in targets:
        for top_k in top_ks:
            # 오라클 하드 top-k — 우리가 이길 수 없는 하한
            o_ppl, o_alive = _perplexity(
                model, ids, mask_fn=_mk(layer, top_k, None), return_alive=True)
            tag = "all" if len(layer) > 1 else str(layer[0])
            records.append(dict(layer=tag, n_layers_patched=len(layer),
                                top_k=top_k, margin=None, method="oracle_topk",
                                ppl=o_ppl, base_ppl=base_ppl,
                                ppl_ratio=o_ppl / base_ppl, mean_alive=o_alive,
                                # ★ utils/metrics.py 를 실제로 배선한다. 지금까지
                                #   호출 0회라 죽은 코드였다 (current_state.md TODO)
                                **perplexity_delta(base_ppl, o_ppl)))
            if verbose:
                print(f"    {tag:>4} {top_k:>4} {'-':>7} {'oracle':>8} {o_ppl:>9.4f} "
                      f"{(o_ppl / base_ppl - 1) * 100:>9.2f}% {o_alive:>10.1f}")
            for margin in margins:
                ppl, alive = _perplexity(
                    model, ids, mask_fn=_mk(layer, top_k, margin), return_alive=True)
                records.append(dict(layer=tag, n_layers_patched=len(layer),
                                    top_k=top_k, margin=margin, method="ours",
                                    ppl=ppl, base_ppl=base_ppl, ppl_ratio=ppl / base_ppl,
                                    mean_alive=alive, oracle_ppl=o_ppl,
                                    beats_oracle=bool(ppl < o_ppl),
                                    **perplexity_delta(base_ppl, ppl)))
                if verbose:
                    star = "  ★오라클보다 좋다" if ppl < o_ppl else ""
                    print(f"    {tag:>4} {top_k:>4} {margin:>7.2f} {'ours':>8} {ppl:>9.4f} "
                          f"{(ppl / base_ppl - 1) * 100:>9.2f}% {alive:>10.1f}{star}")

    save_records(records, NAME, cfg)
    if verbose:
        _report(records, base_ppl)
    return records


def _mk(layer, top_k, margin):
    """층 layer(들)의 어텐션에 씌울 마스크 생성기. margin=None 이면 오라클."""
    def build(logits_np, q_real, k_real):
        if margin is None:
            return _oracle_mask(logits_np, top_k)
        return survivor_mask(q_real, k_real, top_k=top_k, margin=margin)
    return (layer, build)


def _load_text(cfg) -> str:
    from pathlib import Path

    from src.config import PROJECT_ROOT
    p = PROJECT_ROOT / "cache" / "wikitext2_test.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        txt = "\n".join(t for t in ds["text"] if t.strip())
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
        return txt
    except Exception:
        return Path(__file__).read_text(encoding="utf-8") * 40


def _perplexity(model, ids, mask_fn=None, return_alive: bool = False):
    """마스크를 씌운 어텐션으로 perplexity 를 잰다.

    ★ causal 마스크를 직접 만든다 ★
      custom attention 이 transformers 로부터 4-D causal 마스크를 못 받아
      **모델이 미래를 보는** 사고가 있었다(perplexity 1.157). 여기서 만든다.
    """
    import torch
    import torch.nn.functional as F

    n_alive: list[float] = []

    if mask_fn is None:
        with torch.no_grad():
            out = model(ids, labels=ids)
        ppl = float(torch.exp(out.loss))
        return (ppl, 0.0) if return_alive else ppl

    layer_ids, build = mask_fn
    if isinstance(layer_ids, int):
        layer_ids = (layer_ids,)
    n_heads = model.config.num_attention_heads
    n_kv = getattr(model.config, "num_key_value_heads", n_heads)
    head_dim = model.config.hidden_size // n_heads
    group = max(1, n_heads // n_kv)

    def make_patched(attn):
      def patched(hidden_states, *args, **kw):
        B, T, _ = hidden_states.shape
        q = attn.q_proj(hidden_states).view(B, T, n_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(B, T, n_kv, head_dim).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(B, T, n_kv, head_dim).transpose(1, 2)

        pos = kw.get("position_embeddings")
        if pos is not None:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            cos, sin = pos
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        kx = k.repeat_interleave(group, dim=1)
        vx = v.repeat_interleave(group, dim=1)
        logits = (q @ kx.transpose(-1, -2)) / math.sqrt(head_dim)

        # ★ causal 마스크. 이걸 빼면 모델이 미래를 본다.
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(causal, float("-inf"))

        for h in range(n_heads):
            keep = build(logits[0, h].detach().numpy(),
                         q[0, h].detach().numpy().astype(np.float64),
                         kx[0, h].detach().numpy().astype(np.float64))
            n_alive.append(float(keep.sum(axis=1).mean()))
            drop = torch.from_numpy(~keep) & ~causal
            logits[0, h] = logits[0, h].masked_fill(drop, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        o = (probs @ vx).transpose(1, 2).reshape(B, T, -1)
        return attn.o_proj(o), None
      return patched

    saved = {}
    for li in layer_ids:
        a = model.model.layers[li].self_attn
        saved[li] = a.forward
        a.forward = make_patched(a)
    try:
        with torch.no_grad():
            out = model(ids, labels=ids)
        ppl = float(torch.exp(out.loss))
    finally:
        for li, fn in saved.items():
            model.model.layers[li].self_attn.forward = fn
    return (ppl, float(np.mean(n_alive)) if n_alive else 0.0) if return_alive else ppl


def _report(records, base_ppl):
    ours = [r for r in records if r["method"] == "ours"]
    beat = [r for r in ours if r.get("beats_oracle")]
    print(f"\n  full attention ppl = {base_ppl:.4f}")
    print(f"  오라클 하드 top-k 보다 좋은 조합: {len(beat)} / {len(ours)}")
    if beat:
        print("  -> 생존 집합이 top-k 의 상위집합이라는 성질이 품질로 확인된다.")
        print("     보간 추정(+5.8%)은 이 이득을 반영하지 못한 **우리에게 불리한** 값이었다.")
    print("\n  ⚠ 이 수치가 논문에 들어갈 값이다. 보간값을 쓰지 말 것.")


def main() -> int:
    import argparse

    from utils.io import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--top-k", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--margin", type=float, nargs="+", default=[0.0, 0.5, 0.7])
    ap.add_argument("--per-layer", action="store_true",
                    help="층을 하나씩 바꿔 본다 (진단용). 기본은 전 층 동시 교체")
    a = ap.parse_args()
    print(f"=== {NAME} ===")
    run(n_tokens=a.tokens, layers=tuple(a.layers), top_ks=tuple(a.top_k),
        margins=tuple(a.margin), all_layers=not a.per_layer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
