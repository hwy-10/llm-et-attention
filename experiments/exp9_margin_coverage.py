"""EXP9 — ★ margin 스윕 확대: 전 헤드 x 여러 텍스트 ★  (2026-08-29 신설, 팀 2)

`MARGIN = 0.7` 은 **층 8 / 헤드 0 하나**를 wikitext 한 종류로 재서 정한 값이다.
그 값을 레지스터에 박으려면 이런 질문에 답해야 한다.

    (1) 다른 층·헤드에서도 0.7 이 무손실인가
    (2) 다른 텍스트에서도 그런가
    (3) 무손실 knee 가 헤드마다 얼마나 벌어지는가
    (4) 벌어진다면 헤드별 margin 회로가 필요한가

★ (4) 가 진짜 질문이다 ★
  vAttention 계열 연구는 "헤드마다 분포가 달라 하나의 임계값으로는 안 된다"고 한다.
  헤드별 회로를 두면 면적이 32배 든다. 전역 값 하나로 되는지가 설계를 가른다.

  판단 기준:
    전역 안전값     = 모든 (층, 헤드, 텍스트) 조합에서 무손실인 최대 margin
    헤드별 최적합   = 각 조합의 knee 를 개별로 쓴 평균 절감
    이득 = 헤드별 - 전역.  이게 작으면 전역 하나로 간다.

의존성: torch + transformers. 없으면 캐시된 텐서만으로 도는 축소판을 쓴다.
    python -m experiments.exp9_margin_coverage --heads 8 --texts 3
"""

from __future__ import annotations

import numpy as np

from utils.io import save_records

from . import load_config

NAME = "exp9_margin_coverage"

# 서로 성격이 다른 텍스트 — 한 종류만 보면 그 문체에 과적합된다
_TEXTS = {
    "wikitext": None,                       # 캐시 파일에서 (백과사전체)
    "code": (
        "def quantize(x, bits=8):\n"
        "    scale = x.abs().max() / (2 ** (bits - 1) - 1)\n"
        "    return (x / scale).round().clamp(-127, 127), scale\n\n"
        "class BitPlaneCache:\n"
        "    def __init__(self, n_planes, head_dim):\n"
        "        self.planes = [[] for _ in range(n_planes)]\n"
        "        self.head_dim = head_dim\n"
        "    def append(self, k_int8):\n"
        "        for b in range(len(self.planes)):\n"
        "            self.planes[b].append((k_int8 >> b) & 1)\n"
    ) * 12,
    "dialogue": (
        "A: So how does the early termination actually work?\n"
        "B: We process the key bits from the most significant plane down.\n"
        "A: And you can already tell which tokens will lose?\n"
        "B: After a few planes the score is bracketed tightly enough, yes.\n"
        "A: What if two tokens are close?\n"
        "B: Then neither gets dropped. The bound has to clear the threshold.\n"
    ) * 20,
}


def run(cfg=None, verbose: bool = True, n_heads: int = 8, n_layers: int = 4,
        seq_len: int = 512, top_k: int = 16,
        margins=None, texts=None) -> list[dict]:
    cfg = cfg or load_config()
    margins = list(margins or [0.0, 0.4, 0.5, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    texts = list(texts or _TEXTS.keys())

    combos = _capture(cfg, n_layers, n_heads, seq_len, texts, verbose)
    if not combos:
        if verbose:
            print("  캡처할 텐서가 없습니다. torch/transformers 를 설치하거나 "
                  "src.model_hooks 로 먼저 캡처하세요.")
        return []

    records: list[dict] = []
    for (text, layer, head), snap in combos.items():
        # ★ 전처리는 조합당 한 번만. margin 마다 다시 하면 10배 느려진다.
        pre = _prepare(snap)
        for margin in margins:
            m = _measure(pre, top_k=top_k, margin=margin)
            records.append(dict(text=text, layer=layer, head=head, top_k=top_k,
                                margin=margin, **m))
        if verbose:
            print(f"    {text:>10} L{layer:<3} H{head:<3} 완료")

    save_records(records, NAME, cfg)
    if verbose:
        _report(records, margins, verbose)
    return records


def _prepare(snap):
    """양자화 · 비트평면 · 부분내적을 조합당 한 번만 계산한다.

    margin 은 종단 **판정**에만 쓰이므로 여기까지는 margin 과 무관하다.
    (decode_loop 가 스윕을 빠르게 도는 것과 같은 이유다.)
    """
    from src.accumulator import fold_and_quantize_query
    from src.masked_sum import partial_dots
    from src.quantize import quantize_key, to_bitplanes

    q, k = snap
    key = quantize_key(k, bits=8)
    fq = fold_and_quantize_query(q, key)
    planes = to_bitplanes(key.stored, 8)
    partials = partial_dots(fq.stored, planes)          # (8, n_steps, n_tokens)
    exact = fq.stored.astype(np.int64) @ key.stored.astype(np.int64).T
    qs = fq.stored.astype(np.int64)
    return dict(partials=partials, exact=exact,
                q_pos=np.where(qs > 0, qs, 0).sum(axis=-1),
                q_neg=np.where(qs < 0, qs, 0).sum(axis=-1),
                n=q.shape[0])


def _measure(pre, *, top_k: int, margin: float) -> dict:
    """전처리된 조합에서 이 margin 의 절감과 보존율."""
    from src.bounds import StepBounds
    from src.memory import latency_planes
    from src.terminator import run_step
    from src.threshold import ThetaPolicy

    pol = ThetaPolicy(top_k=top_k, margin=margin, margin_mode="relative_width")
    n = pre["n"]
    read_live = read_all = kept = total = 0

    for s in range(max(top_k, 32), n):
        n_act = s + 1
        b = StepBounds(q_pos=int(pre["q_pos"][s]), q_neg=int(pre["q_neg"][s]))
        res = run_step(pre["partials"][:, s, :n_act], b, pol,
                       decision_latency=latency_planes(n_act, 32, 8))

        read_live += int(res.read_live.sum())
        read_all += 8 * n_act

        true = pre["exact"][s, :n_act]
        kk = min(top_k, n_act)
        # ★ 동점 안전. 값으로 비교한다 — decode_loop 와 같은 정의
        thr = float(np.partition(true, -kk)[-kk])
        alive_true = np.where(res.alive, true, -np.inf)
        picked = np.argpartition(alive_true, -kk)[-kk:]
        kept += int(np.count_nonzero(alive_true[picked] >= thr))
        total += kk

    return dict(read_saving=1.0 - read_live / read_all if read_all else 0.0,
                retention=kept / total if total else 1.0)


def _cache_path(cfg, text: str, layer: int, head: int, seq_len: int):
    from src.config import PROJECT_ROOT
    name = cfg.get("model.model.name", "model")
    return (PROJECT_ROOT / "cache" / "coverage"
            / f"{name}_{text}_L{layer}_H{head}_T{seq_len}.npz")


def _capture(cfg, n_layers, n_heads, seq_len, texts, verbose):
    """(텍스트, 층, 헤드) -> (q, k).

    ★ 캐시를 먼저 본다 ★
      모델을 올리는 데 RAM 이 많이 든다(fp32 약 5GB). 이 기계에서 여유가 6GB 밖에
      없어 로딩 중 세그폴트가 반복됐다. 캐시가 다 있으면 **모델을 아예 안 올린다.**
      캐시를 만들려면 --capture 로 한 번 돌린다 (모델을 텍스트당 1회만 로드).
    """
    total = int(cfg.get("model.model.n_layers", 16))
    layer_ids = [int(round(i * (total - 1) / max(1, n_layers - 1)))
                 for i in range(n_layers)]

    out, missing = {}, []
    for name in texts:
        for L in layer_ids:
            for H in range(n_heads):
                p = _cache_path(cfg, name, L, H, seq_len)
                if p.exists():
                    d = np.load(p)
                    out[(name, L, H)] = (d["q"], d["k"])
                else:
                    missing.append((name, L, H))

    if out and verbose:
        print(f"  캐시에서 {len(out)}조합")
    if not missing:
        return out

    if verbose:
        print(f"  캐시 없는 {len(missing)}조합 — 모델을 올립니다 "
              f"(RAM 부족이면 --capture 로 따로 돌리세요)")
    fresh = _capture_from_model(cfg, layer_ids, n_heads, seq_len,
                               sorted({t for t, _l, _h in missing}), verbose)
    out.update(fresh)
    return out


def _capture_from_model(cfg, layer_ids, n_heads, seq_len, texts, verbose, save=True):
    """모델을 **텍스트당 한 번만** 올려 전 층·전 헤드를 한꺼번에 뽑는다.

    model_hooks.capture_qk 는 호출마다 가중치를 다시 읽는다. 조합이 96개면
    로드를 96번 하게 되어 실행이 죽는다. 여기서는 훅을 모든 층에 한꺼번에
    걸고 forward 한 번으로 받는다.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    except ImportError:
        if verbose:
            print("  torch / transformers 가 없습니다.")
        return {}

    hf_id = cfg.get("model.model.hf_id", "unsloth/Llama-3.2-1B")
    tok = AutoTokenizer.from_pretrained(hf_id)
    # ★ bfloat16 으로 올린다. 여기서 뽑은 q/k 는 곧바로 INT8 로 양자화되므로
    #   bf16 정밀도면 충분하고, fp32 대비 메모리가 절반이다.
    kw = dict(low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, **kw)
    model.eval()

    n_h = int(model.config.num_attention_heads)
    n_kv = int(getattr(model.config, "num_key_value_heads", n_h))
    head_dim = model.config.hidden_size // n_h
    group = max(1, n_h // n_kv)

    out = {}
    for name in texts:
        body = _TEXTS.get(name) or _wikitext(cfg)
        grabbed, hooks = {}, []
        for L in layer_ids:
            attn = model.model.layers[L].self_attn
            hooks.append(attn.q_proj.register_forward_hook(
                lambda m, i, o, L=L: grabbed.__setitem__(("q", L), o.detach())))
            hooks.append(attn.k_proj.register_forward_hook(
                lambda m, i, o, L=L: grabbed.__setitem__(("k", L), o.detach())))
        try:
            ids = tok(body, return_tensors="pt", truncation=True, max_length=seq_len)
            with torch.no_grad():
                model(**ids)
        finally:
            for h in hooks:
                h.remove()

        n_tok = int(ids["input_ids"].shape[1])
        pos = torch.arange(n_tok).unsqueeze(0)
        for L in layer_ids:
            q = grabbed[("q", L)][0].reshape(-1, n_h, head_dim)
            k = grabbed[("k", L)][0].reshape(-1, n_kv, head_dim)
            # 모델과 같은 순서로 RoPE. (T,H,D) -> (1,H,T,D) -> 되돌리기
            q4 = q.permute(1, 0, 2).unsqueeze(0)
            k4 = k.permute(1, 0, 2).unsqueeze(0)
            cos, sin = model.model.rotary_emb(q4, pos)
            q4, k4 = apply_rotary_pos_emb(q4, k4, cos, sin)
            q = q4[0].permute(1, 0, 2)
            k = k4[0].permute(1, 0, 2)
            for H in range(min(n_heads, n_h)):
                kv = H // group               # GQA — model_hooks 와 같은 규칙
                qa = np.ascontiguousarray(q[:, H, :].float().numpy(), dtype=np.float64)
                ka = np.ascontiguousarray(k[:, kv, :].float().numpy(), dtype=np.float64)
                out[(name, L, H)] = (qa, ka)
                if save:
                    p = _cache_path(cfg, name, L, H, seq_len)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(p, q=qa, k=ka)
        grabbed.clear()
        if verbose:
            print(f"  캡처 {name}: {n_tok}토큰 x 층 {layer_ids} x 헤드 {min(n_heads, n_h)}개")
    del model
    return out


def capture_all(cfg=None, n_layers=4, n_heads=8, seq_len=512, texts=None, verbose=True):
    """캐시만 만들고 끝낸다. RAM 이 빠듯할 때 측정과 분리해서 돌린다."""
    cfg = cfg or load_config()
    total = int(cfg.get("model.model.n_layers", 16))
    layer_ids = [int(round(i * (total - 1) / max(1, n_layers - 1)))
                 for i in range(n_layers)]
    got = _capture_from_model(cfg, layer_ids, n_heads, seq_len,
                              list(texts or _TEXTS), verbose)
    if verbose:
        print(f"  캐시 {len(got)}조합 저장 -> cache/coverage/")
    return len(got)


def _wikitext(cfg) -> str:
    from src.config import PROJECT_ROOT
    p = PROJECT_ROOT / "cache" / "wikitext2_test.txt"
    return p.read_text(encoding="utf-8") if p.exists() else "The quick brown fox. " * 900


def global_safe_margin(records, tol: float = 1.0) -> float:
    """모든 조합에서 보존율 >= tol 인 최대 margin. 이게 레지스터에 박을 값이다."""
    ms = sorted({r["margin"] for r in records})
    best = 0.0
    for m in ms:
        rows = [r for r in records if r["margin"] == m]
        if rows and min(r["retention"] for r in rows) >= tol:
            best = m
    return best


def per_head_knee(records, tol: float = 1.0) -> dict:
    """조합마다의 무손실 knee."""
    out = {}
    for r in records:
        key = (r["text"], r["layer"], r["head"])
        if r["retention"] >= tol:
            out[key] = max(out.get(key, 0.0), r["margin"])
    return out


def _report(records, margins, verbose):
    g = global_safe_margin(records)
    knees = per_head_knee(records)
    combos = sorted({(r["text"], r["layer"], r["head"]) for r in records})

    print(f"\n  조합 {len(combos)}개 x margin {len(margins)}점 = {len(records)}회 측정")

    print("\n  margin 별 — 전 조합의 최악값이 중요하다")
    print(f"    {'margin':>7} {'절감 평균':>10} {'절감 최소':>10} "
          f"{'보존 평균':>10} {'보존 최악':>10}  판정")
    for m in margins:
        rows = [r for r in records if r["margin"] == m]
        if not rows:
            continue
        sv = [r["read_saving"] for r in rows]
        rt = [r["retention"] for r in rows]
        print(f"    {m:>7.2f} {np.mean(sv) * 100:>9.1f}% {min(sv) * 100:>9.1f}% "
              f"{np.mean(rt):>10.5f} {min(rt):>10.5f}  "
              f"{'무손실' if min(rt) >= 1.0 else ''}")

    print(f"\n  ★ 전역 안전 margin = {g:.2f}  (모든 조합에서 보존율 1.0)")

    if knees:
        vals = list(knees.values())
        print(f"    조합별 knee   최소 {min(vals):.2f}  최대 {max(vals):.2f}  "
              f"중앙 {np.median(vals):.2f}")
        # 헤드별 회로를 두면 얼마나 더 버나
        gs = [r["read_saving"] for r in records if r["margin"] == g]
        per = [max((r["read_saving"] for r in records
                    if (r["text"], r["layer"], r["head"]) == c
                    and r["margin"] == knees[c]), default=0.0)
               for c in combos if c in knees]
        if gs and per:
            gain = (np.mean(per) - np.mean(gs)) * 100
            print(f"\n  ★ 헤드별 margin 회로의 이득 = {gain:+.1f}%p")
            print(f"     전역 {g:.2f}  ->  절감 {np.mean(gs) * 100:.1f}%")
            print(f"     헤드별 최적    ->  절감 {np.mean(per) * 100:.1f}%")
            print("     " + ("이득이 작다. 전역 값 하나로 간다 (면적 32배를 아낀다)."
                             if gain < 5 else
                             "★ 이득이 크다. 헤드별 margin 레지스터를 검토할 것."))

    worst = sorted(records, key=lambda r: r["retention"])[:3]
    if worst and worst[0]["retention"] < 1.0:
        print("\n  가장 취약한 조합")
        for r in worst:
            print(f"    {r['text']:>10} L{r['layer']:<3} H{r['head']:<3} "
                  f"margin {r['margin']:.2f}  보존 {r['retention']:.5f}")


def main() -> int:
    import argparse

    from utils.io import enable_utf8_stdout
    enable_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--texts", type=str, nargs="+", default=list(_TEXTS))
    ap.add_argument("--capture", action="store_true",
                    help="캡처만 하고 끝낸다. RAM 이 빠듯할 때 측정과 분리")
    a = ap.parse_args()
    print(f"=== {NAME} ===")
    if a.capture:
        capture_all(n_layers=a.layers, n_heads=a.heads,
                    seq_len=a.seq_len, texts=a.texts)
        return 0
    run(n_heads=a.heads, n_layers=a.layers, seq_len=a.seq_len,
        top_k=a.top_k, texts=a.texts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
