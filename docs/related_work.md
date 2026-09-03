# 관련 연구 조사 — 2026-08 기준

이 문서는 프로젝트가 인용하던 4편([1] LeOPArd, [2] SpAtten, [4] BitStopper, [11] KIVI)을
기준으로 **2023~2026년 최신 문헌**을 다시 조사한 결과다.

조사 방법: 5개 각도 병렬 웹 검색 → 22개 primary source 원문 fetch →
110개 검증 가능한 주장 추출 → 상위 25개에 대해 **3표 적대적 검증**(2표 이상 반박이면 기각).
**14건 확정, 11건 기각.** 기각된 것도 아래 §7에 전부 적어 두었다 — 쓰면 안 되는 주장이기 때문이다.

> ⚠ **시효**: 근거의 절반이 2025-12 ~ 2026-04 문헌이고, 아래 §1의 Tsinghua 그룹은
> **약 6개월 주기로 자기 계승 논문**을 내고 있다. 투고 직전에 반드시 재검색할 것.

---

## 0. 먼저 조치해야 할 세 가지

### ★ (1) 프로젝트 전제 하나가 반증되었다 — 문구 수정 필수

지금까지 이렇게 서술해 왔다.

> K는 **asymmetric unsigned** 여야 한다. two's-complement signed 면 MSB 자리값이
> −128 이라 **상한식이 깨진다.**

**뒷문장이 틀렸다.** PADE 와 BitStopper 두 편이 각각 **two's-complement signed 정수에서
유효한 bound 를 유도**한다. 이유는 MSB-first 순서 자체에 있다 — 부호 비트(가중치 −2^(p−1))는
**라운드 0에서 이미 확정**되므로 항상 *결정된* 부분합 S_m 에 들어가고, *미확정* 나머지 비트의
가중치는 전부 양수다. 프로젝트의 R_m = (2^(8−m) − 1)·Q+ 자체가 이미 Q의 부호로 조건화된
식이라, m ≥ 1 구간에서는 signed K 에 그대로 전이된다.

정확한 서술은 이렇다.

> asymmetric unsigned 를 쓰면 **8개 plane 전부에 단일 R_m 공식**이 적용되어
> sign plane 특수 처리와 양방향 구간 로직이 불필요해진다. **하드웨어 단순화를 위한
> 설계 선택**이지, 수학적 필연이 아니다.

설계 결정 자체는 유지해도 된다. 근거만 "필연" → "편의"로 낮추면 된다.
`config/quant.yaml`, `src/quantize.py` 주석, `architecture.md` §2 를 함께 고쳐야 한다.

### ★ (2) 신규성 주장을 좁혀야 한다

**PADE (HPCA 2026)** 가 이 프로젝트의 핵심 메커니즘을 사실상 그대로 구현했다.
"부분합 bound 에 의한 bit-level 조기 종료"는 더 이상 신규 기여가 아니다.
남는 차별점은 세 가지뿐이다 — **(a) FPGA / DSP-free 구현, (b) BRAM word-width 와
work-compaction, (c) INT8 asymmetric unsigned 데이터패스.**

### ★ (3) [4] BitStopper 인용 수치가 틀렸다

"~6.9% control-logic overhead" 는 **area 가 아니라 power** 수치이고, 조기 종단 관련
블록 전체를 포함하지도 않는다. 상세는 §1-4.

---

## 0.1 ★★ 차별화 문장 (2026-08-29 작성, 측정 근거 포함) ★★

`docs/related_work.md` 초판은 남는 차별점을 세 가지로 좁혔다 —
**(a) FPGA/DSP-free, (b) BRAM word-width + work-compaction, (c) INT8 unsigned 데이터패스.**
그 뒤 골든모델이 완성되면서 **(a)~(c) 를 숫자로 뒷받침할 수 있게 됐고, 넷째가 추가됐다.**

### ★ (d) θ 의 정의가 다르다 — 우리는 무손실이 증명된다

이것이 초판이 놓친 가장 큰 차별점이다.

```
   PADE Eq.4     T = max(S^{:,min}) - alpha x radius        (0 <= alpha <= 1, radius 5)
   BitStopper    eta = max(A^r) - alpha x radius            (alpha ~ 0.6 이 knee)
   BETA          MTS (max-based threshold selection)
   MCBP          radius 3, 적응 계수 alpha_r

   우리          theta = 활성 토큰 하한 중 **k번째로 큰 값**
```

**계보 전체가 `max(하한) - alpha x radius` 를 쓴다.** 이건 1등 하한을 기준선으로 삼고
휴리스틱하게 느슨하게 푸는 것이라, **top-k 보장이 나오지 않는다.** 그래서 PADE 자신도
무손실이라고 하지 않고 *"standard (0% loss), aggressive (1% loss)"* 두 설정을 둔
**정확도 노브**로 제시한다.

우리는 **k번째로 큰 하한**을 쓴다. 그러면 두 줄로 증명된다.

```
   (1) 하한이 theta 이상인 토큰이 최소 k개 있다. 그 참 점수는 전부 theta 이상이다.
   (2) 상한이 theta 보다 작은 토큰은 참 점수도 theta 보다 작다.
       -> 위 k개보다 확실히 작으므로 상위 k 에 들 수 없다.
```

**근사가 아니라 정확하다.** 측정으로도 확인된다.

```
   상한식 L_m <= s <= U_m     16,785,408 건 검사, 위반 0 건     (실제 Llama T=2048)
   top-16 보존율              T = 512/1024/2048 전부 1.00000
   384조합(3텍스트x4층x32헤드) margin 0.40 까지 보존율 1.00000  (0.40 에서 수렴)
```

> **주의 — 초판의 "무손실로 표현하면 안 된다" 는 `max - alpha x radius` 를 채택했을
> 때의 경고였다.** 우리는 그 정책을 채택하지 않았으므로 경고가 적용되지 않는다.
> 다만 **margin > 0 (근사 모드) 는 무손실이 아니다.** 그 구간은 곡선으로 보고한다.

`margin` 은 PADE 의 `alpha` 와 같은 역할이지만 **위에 얹는 선택지**이지 기본이 아니다.
기본값 0.40 은 384조합에서 보존율 1.0 이고, 0.70 은 0.99505 다.

### ★ (a) FPGA / DSP-free — 이제 숫자가 있다

PADE 는 TSMC 28nm ASIC (Synopsys DC + CACTI/Ramulator) 이고 **FPGA 평가가 없다.**
FPGA 는 SRAM 이 아니라 **BRAM 블록**이라 제약이 다르고, 그 차이가 결과를 바꾼다.

```
   DSP 슬라이스           0개      (masked_sum = AND + 가산트리, 누산 = 시프트)
   연산축 (2)/(1)         8.00     비트평면 순차의 구조적 대가
   메모리 포함 (3)/(1)    0.743    실제 Llama T=2048, n_ports=32
   실효 speedup vs (1)    1.111    Fmax 저하 165/200 반영
   손익분기               T ~ 1,280
```

**"연산만 세면 8배 불리한데 메모리를 넣으면 이긴다"** 는 것 자체가 FPGA 평가에서만
나오는 결론이다. ASIC 논문은 이 축을 다르게 본다.

### ★ (b) BRAM word-width + work-compaction — ASIC 에는 없는 문제

FPGA BRAM 은 워드 단위로 읽는다. **워드 안에 살아있는 토큰이 하나라도 있으면 워드
전체를 읽어야 한다.** 종단된 토큰이 흩어져 있으면 절감이 실현되지 않는다.

```
   워드폭   batch          compaction     (실제 Llama T=2048, 이론 35.6%)
      1    28.7% (81%)    35.6% (100%)
     32    23.2% (65%)    35.1%  (99%)
     64    19.4% (54%)    34.5%  (97%)
```

그리고 **병목 위치가 종단의 가치를 정한다** — `total = max(compute, memory)` 에서
`max()` 가 연산을 집으면 메모리 절감이 시간으로 환산되지 않는다.

```
   포트 |   T=512        T=1024       T=2048
        | 시간   병목  | 시간   병목  | 시간   병목
     16 | .760 메모리  | .713 메모리  | .668 메모리
     32 | .990  연산   | .836  연산   | .733  연산

   교차점    T=512: 24     T>=1024: 32
```

**이 두 표는 계보 어디에도 없다.** ASIC SRAM 은 포트 폭을 자유롭게 정하므로
애초에 문제가 되지 않는다.

### ★ (c) INT8 asymmetric unsigned — 성립 조건을 명시한다

계보는 signed 정수를 쓰고, 그러면 MSB 가 부호비트라 `2^b` 가중이 성립하지 않는다.
우리는 unsigned 저장 + zero-point 보정을 쓴다. 상한식의 성립 조건은 하나다.

```
   0 <= K_stored < 2^n_planes
```

벗어나면 비트평면 변환이 **조용히 잘라내므로** 평면이 K 를 대표하지 못한다
(`K=256 -> 복원값 0`). 골든모델은 이 조건을 입구에서 예외로 막는다.

### ★ (e) 판정 지연이 문맥 길이의 함수라는 것 — 계보에 서술이 없다

되먹임 루프에 지연이 있다. 평면 `m` 의 판정이 나올 때 평면 `m+1` 의 읽기는 이미
나가 있다. **그 지연이 몇 "평면" 인지는 T 가 정한다.**

```
   판정 지연(평면) = ceil(파이프라인 사이클 / ceil(T / LANES))

    T      지연(평면)    절감(고정 1 가정)   절감(실제)
  128          2            14.9%            4.0%
  256          1            19.4%           11.4%
  512          1            23.3%           21.4%
```

**짧은 문맥에서 절감이 거의 사라진다.** 이 보정을 넣지 않으면 최대 2배 과대평가한다.
계보 논문 어디에도 이 환산이 명시되지 않았다 — ASIC 은 파이프라인이 얕아
문제가 덜 드러났을 가능성이 있다.

### ★ (f) 생존 집합이 하드 top-k 보다 좋다는 것을 실측했다

정확 모드의 생존 집합은 top-k 의 **상위집합**이다. 상한이 theta 를 못 넘는 것만
버리므로 "버릴 수 없는 토큰" 이 함께 남는다. 종단 로직을 어텐션에 직접 넣어
**전 층(16개) 동시 교체**로 perplexity 를 쟀다.

```
    k   방식              ppl    full 대비   평균 생존
    8   오라클 top-k    8.3313    +9.13%       7.9
    8   우리 (정확)     8.0996    +6.09%      10.0  ★
   16   오라클 top-k    7.9114    +3.63%      15.8
   16   우리 (정확)     7.8464    +2.78%      19.5  ★
   16   우리 (m=0.70)   7.8967    +3.43%      16.8  ★
   32   오라클 top-k    7.7491    +1.50%      31.0
   32   우리 (정확)     7.7123    +1.02%      37.3  ★

   full attention ppl = 7.6345      ★ = 오라클보다 좋다 (6/6 조합)
```

**"top-k 를 쓰는 어떤 방법보다 낫다"** 를 수치로 말할 수 있다. 6개 조합 전부에서
하드 top-k 오라클을 이긴다 — 상위집합 효과가 우연이 아니라는 뜻이다.

> PADE 계보는 이 비교를 하지 않는다. `max(하한) - alpha x radius` 는 top-k 집합을
> 만들지 않으므로 "하드 top-k 대비" 라는 기준선 자체가 성립하지 않기 때문이다.

---

## 0.2 논문에 쓸 한 문단

> 본 연구는 MSB-우선 비트평면 어텐션의 조기 종단을 **FPGA 에서 특성화한다.**
> 메커니즘 자체는 PADE(HPCA 2026) 계보가 ASIC 에서 선행했으나, 세 가지가 다르다.
> **첫째, 기준선 정의가 다르다** — 계보는 `max(하한) − α·radius` 라는 휴리스틱을 쓰고
> 무손실을 주장하지 않는 반면, 본 연구는 **k번째로 큰 하한**을 써 top-k 무손실을
> 증명하고 1,678만 건의 상한식 검사와 384개 층·헤드·텍스트 조합에서 확인한다.
> **둘째, FPGA BRAM 의 워드 단위 읽기가 절감의 실현률을 좌우한다** — 이론 절감의
> 54~100% 만 실현되며, 워드폭과 스케줄 정책이 함께 정한다. **셋째, 병목 위치가
> 종단의 가치를 정한다** — 포트 수가 교차점(문맥 512 에서 24, 1024 이상에서 32)을
> 넘으면 메모리 절감이 시간으로 환산되지 않는다. 이 세 가지는 SRAM 포트 폭을 자유롭게
> 정하는 ASIC 평가에서는 드러나지 않는다.


## 1. Angle 1 — Bit-serial / MSB-first 조기 종단 가속기

### 1-1. 연구 지형: LeOPArd/SpAtten 이후는 공백이 아니었다

2025~2026 년의 이 분야는 **Tsinghua 대학 Yin/Hu 그룹 단일 계보**가 주도하고 있다.
1저자 Huizheng Wang, 2저자 Hongbin Wang, 교신저자 Yang Hu · Shouyi Yin 이 네 편에 공통이다.

| 시기 | 논문 | 위치 |
|---|---|---|
| 2025-08 | **BETA** (TCAS-II) | predictor 기반 bit-grained filter |
| 2025-09 | **MCBP** (MICRO 2025) | bit-grained progressive prediction, radius=3 |
| 2025-12 | **BitStopper** (preprint) | stage-fusion + early termination |
| 2026-02 | **PADE** (HPCA 2026) | predictor 제거, 순수 bound 기반 |

서로를 인용하며 앞 편의 한계를 다음 편의 동기로 삼는 구조다.
**"LeOPArd/SpAtten 이후 공백"이라는 서술은 성립하지 않는다.**
[1]과 [2]는 이제 SOTA 가 아니라 계보의 출발점 / baseline 이다.

### 1-2. ★ PADE — 가장 가까운 선행연구 (최우선 정독)

> Huizheng Wang, Hongbin Wang, Zichuan Wang, Zhiheng Yue, Yang Wang, Chao Li,
> Yang Hu, Shouyi Yin. **"PADE: A Predictor-Free Sparse Attention Accelerator via
> Unified Execution and Stage Fusion."** HPCA 2026.
> arXiv:[2512.14322](https://arxiv.org/abs/2512.14322) · IEEE Xplore doc 11408448
> (공식 프로그램 Session "Efficient LLM Inference Techniques", 2026-02-02)

**하는 일:** BUI-GF (bit-wise uncertainty interval 기반 guard filtering),
BS-OOE (bidirectional sparsity out-of-order execution), ISTA (interleaving-based
sparsity-tiled attention).

원문에서 확인된 일치점이 불편할 정도로 많다.

* MSB-first 명시 — *"Start with the first bit plane (i.e., MSB) of Keys for
  bit-serial speculating of Q×K^T"*
* 부분합에 구간을 씌움 — Eq.3 의 `S^{r,min} = S^r + I^{r,min}`, `S^{r,max} = S^r + I^{r,max}`
* 종료 판정식 — Decision Unit 이 *"checks whether S_{i,j}^r + I_i^{r,max} > T_i"*.
  참이면 다음 bit plane 요청, 거짓이면 scoreboard 에서 token 축출
* **운용점까지 동일** — INT8, Key 벡터당 8개 1-bit plane, Q 는 full 8-bit 병렬 /
  K 만 bit-serial
* Prefill 과 decode 모두 다룸

**차이점:** 평가가 TSMC 28nm ASIC (Synopsys DC + CACTI/Ramulator) 이고 **FPGA 평가가 없다.**
H100 대비 7.43x speedup / 31.1x energy efficiency. bound 도 양방향 구간이라
프로젝트의 단측 R_m 과 형태가 다르다. SpAtten 은 인용하지만 LeOPArd 는 인용하지 않는다.

**→ ADOPT + 신규성 재정의.** 논문 작성 전 반드시 전문 정독.

### 1-3. BETA — SpAtten[2]을 정량적으로 추월한 peer-reviewed 후속

> Huizheng Wang, Hongbin Wang, Zhiheng Yue, Jingyao Liu, Taiquan Wei, Shaojun Wei,
> Yang Hu, Shouyi Yin. **"BETA: A Bit-Grained Transformer Attention Accelerator
> With Efficient Early Termination."** IEEE TCAS-II, vol. 72, no. 10, pp. 1433–1437, Oct 2025.
> DOI [10.1109/TCSII.2025.3596228](https://doi.org/10.1109/TCSII.2025.3596228)

Abstract 원문: *"BETA achieves 5.4x, 6.5x, 1.8x improvements in energy efficiency
than the state-of-the-art Transformer accelerators Sanger, **Spatten** and SOFA."*
구성: bit-grained multi-round filter (BMF), max-based threshold selection (MTS),
bit-wise out-of-order execution (BOOE).

**중요한 한정:** BitStopper 가 BETA 를 [34]로 인용하며 비판하기를, BETA 는
*"likewise retains an additional sparsity predictor"* 이고 *"the absence of a
bit-uncertainty margin renders its pruning decisions highly inaccurate"* —
즉 **건전한 상한 bracket 이 아니라 예측기**다.

**→ CITE AS RELATED WORK.** [2] SpAtten 이 추월당했음을 명시하되,
**R_m bound 의 prior art 로는 인용하지 말 것** (성격이 다르다).

> 🚨 **이름 충돌 경고**
> "BETA: **Binarized** Energy-Efficient Transformer Accelerator at the Edge"
> (arXiv:2401.11851) 은 Xilinx ZCU102 / Vivado 2022.2 **FPGA** 설계로 완전히 다른 논문이다.
> 검색엔진이 둘을 섞는다. ZCU102 수치를 TCAS-II BETA 에 붙이면 안 된다.

### 1-4. ★ BitStopper — 인용 정정 필요

> Huizheng Wang, Hongbin Wang, Shaojun Wei, Yang Hu, Shouyi Yin.
> **"BitStopper: An Efficient Transformer Attention Accelerator via Stage-fusion
> and Early Termination."** arXiv:[2512.06457](https://arxiv.org/abs/2512.06457) (2025-12-06).
> **미심사 preprint** — v1 단독, journal-ref/venue 필드 없음.

원문은 이렇게 되어 있다.

> *"The newly added Bit Margin Generator and LATS modules ... incurring only
> **4.9% area and 6.9% power** overhead. Additionally, the integration of the
> Scoreboard and Pruning Engine into PE lanes enables stage fusion, adding
> **5.8% area and 4.9% power** overhead."*

| 프로젝트의 현재 인용 | 실제 |
|---|---|
| "control-logic overhead 6.9%" | 6.9% 는 **power**. area 는 4.9% |
| (단일 수치로 서술) | 블록이 **두 그룹**. 합치면 대략 **10~12%** |

* 합계 10.7% / 11.8% 는 **논문에 인쇄된 값이 아니라 단순 합**이다. 기준선이 달라
  엄밀한 가산성이 없으므로 "약 10~12% 수준" 정도로 쓰고 근거를 밝힐 것.
* 기준 설계는 TSMC 28nm, 1GHz, 6.84 mm², 703 mW, peak 11.36 TOPS/W.
* **양자화가 INT12 per-tensor** 다 (프로젝트는 INT8 → plane 수가 다름).
* 일부 검색엔진이 "HPCA 2025 게재"로 표시하는 것은 **무관한 저자 프로필 링크에서 온 오류**다.
  preprint 로 표기할 것.

**→ CITE-CORRECTION 필수.** control-logic 오버헤드 예산을 6.9% 에 맞춰 세웠다면 상향 조정.

### 1-5. MCBP (차순위)

> **MCBP** (MICRO 2025). arXiv:[2509.10372](https://arxiv.org/abs/2509.10372) ·
> DOI 10.1145/3725843.3756037

동일 계보. bit-grained progressive prediction + radius(기본값 3) 기반
threshold updating module. §2 의 θ 갱신 기구 원형이다.

---

## 2. Angle 2 — θ(종단 임계값)를 어떻게 고르는가

### 2-1. ★ 이 계보의 공통 답: running max of lower bounds − α × radius

네 편이 **일관되게 같은 답**을 내놓는다. θ 는 oracle 도, 고정 상수도 아니고,
**bit-plane 라운드마다 온라인 재추정**한다.

```
θ = max(현재까지의 lower bound들) − α × radius        (0 ≤ α ≤ 1, radius 기본값 5)
```

* **PADE** Eq.4 원문: *"T = max(S_{i,:}^{:,min}) − α × radius, 0 ≤ α ≤ 1"* +
  *"the scores are not confined to a specific bit plane, but are derived from
  all current processed bit planes. Based on our experiments, we set the default
  radius to 5."* 전용 Threshold Updating Module 이 실행 중 T_i 를 재생성해
  row i 의 모든 PE lane 에 broadcast.
* **BitStopper** Eq.3 원문: *"η_i = max(A_{i,:}^r) − α × radius, α ∈ [0,1]"*.
  α 0.2~0.8 스윕에서 *"when α falls below 0.6, the reduction in complexity begins
  to plateau, while 1/PPL drops sharply"* → **α ≈ 0.6 이 knee**.
* **BETA** 는 같은 것을 MTS (max-based threshold selection) 라고 부른다.
* **MCBP** 는 radius 기본값 3, 적응 계수 α_r.

**중요:** PADE 는 이것을 무손실이라고 하지 않는다. *"standard (0% loss),
aggressive (1% loss)"* 두 설정을 두고 α 를 0.1 단위로 스윕하는 **정확도 노브**다.

**→ ADOPT / BENCHMARK.** θ 를 고정 상수로 두려던 계획은 폐기하고,
running-max-of-lower-bounds − α·radius 레지스터를 기본 구현으로 삼되
프로젝트의 단측 bound (S_m ≤ s ≤ S_m + R_m) 에 맞게 각색할 것.
**단 "무손실"로 표현하면 안 된다.**

### 2-2. ★ `prev_step` 정책 — 문헌에 답이 없다

프로젝트가 발견한 것 — **이전 decode step 의 θ 재사용은 무손실이 아니다**
(`src/threshold.py`, `tests/test_decode_loop.py`) — 에 대해:

* 이 계보 **어디에도 step 간 θ 재사용 메커니즘이 등장하지 않는다.**
  전부 라운드별 online 재추정만 쓴다.
* "verify 단계를 붙이면 무손실이 된다"는 주장을 검증했으나 **1-2 로 기각**되었다.
* **published 분석을 찾지 못했다.**

**→ 프로젝트의 열린 질문으로 남는다.** 동시에 작은 기여가 될 수 있다 —
자체 실험으로 반례를 이미 갖고 있기 때문이다.

### 2-3. Salca — 히스토그램 기반 O(n) top-k (FPGA 친화적)

> Wang Fan, Wei Cao, Xi Zha, Kedi Ma, MingQian Sun, Jialin Chen, Fengzhe Zhang, Fan Zhang.
> **"Salca: A Sparsity-Aware Hardware Accelerator for Efficient Long-Context
> Attention Decoding."** arXiv:[2604.24820](https://arxiv.org/abs/2604.24820) (2026-04-27).

정렬도 running top-k 레지스터도 아닌 **하드웨어 히스토그램**으로 임계값을 구한다.
FP16 score 를 INT8 로 양자화해 **256-entry 카운터 배열의 주소**로 쓰고,
높은 주소부터 누적해 누적합이 K 를 넘는 첫 주소를 근사 임계값으로 삼는다.
필터 복잡도 O(n log k) → **O(n)**, 64-way 병렬에서 **2n/64 cycle**.

**한정:** 이미 계산이 끝난 FP16 score 에 적용되는 **post-Q·K^T 필터**이지,
bit-plane 라운드별 종료용 θ 생성기가 아니다.

**→ 선택적 ADOPT.** 종료 이후 top-k 압축 단계를 별도로 둔다면, 256-entry 히스토그램은
정렬 네트워크 대비 FPGA LUT/BRAM 에 매우 잘 맞는다.

---

## 3. Angle 3 — K 캐시 양자화 (KIVI[11] 이후)

**이 각도는 커버리지가 얕다.** 확정된 것은 Titanus 한 건뿐이고,
가장 중요한 질문(§8-2)은 검증하지 못했다.

### 3-1. Titanus — KIVI 방식의 하드웨어 비효율을 명명

> Peilin Chen, Xiaoxuan Yang. **"Titanus: Enabling KV Cache Pruning and
> Quantization On-the-Fly for LLM Acceleration."** GLSVLSI 2025.
> arXiv:[2505.17787](https://arxiv.org/abs/2505.17787)

KIVI 가 대중화한 per-channel K 양자화의 문제를 **NiPCQ**
(Non-independent Per-Channel Quantization) 로 명명한다 — channel 이 token 축을
가로지르므로 *"the need to re-quantize all channels whenever a newly generated
token arrives."* 해법 **HQE** 는 채널을 여러 level 로 나누고 상위 level 의
tolerance range 를 넓혀 각 token 이 정확히 한 번만 양자화되게 한다.
반응형이다 — 첫 범위 초과 시 상위 level 생성.

**→ CITE AS RELATED WORK, NO ACTION.** per-channel asymmetric 을 유지한 채
유지비용만 낮추므로 signed/rotated 표현을 도입하지 않는다.
**R_m 의 전제를 깨지 않는다.** [11] KIVI 의 2025 년 후속으로 각주 처리하면 충분하다.

> 저장 오버헤드 "무시할 만함"은 저자 자체 평가다. 정량치는
> *"less than 13 levels per channel on average"* (context 1024) 뿐이니
> 그대로 인용하지 말고 수치와 함께 쓸 것.

---

## 4. Angle 4 — FPGA / DSP-free

### ★ 이 각도는 사실상 공백이다

검증된 2025~2026 bit-serial 조기 종단 attention accelerator 는 **전부 ASIC** 이다.

| 논문 | 평가 플랫폼 |
|---|---|
| PADE | Synopsys DC / TSMC 28nm + CACTI · Ramulator |
| BitStopper | Synopsys DC / 28nm, 6.84 mm², 703 mW |
| BETA | baseline 3종(Sanger/SpAtten/SOFA) 전부 ASIC, 28nm 정규화 관행 |
| Salca | ASIC, 비교 대상 A100 GPU |

다음 두 질문을 **지지하거나 반박하는 검증된 문헌을 확보하지 못했다.**

1. 최신 Xilinx 부품에서 "DSP 를 피하고 LUT 를 쓰는 것"이 iso-area 로 유리한가
2. BRAM word-width 대 sparsity 실현 — work-compaction 이 필수라는 보고가 다른 데도 있는가

**→ 기회이자 리스크.**
**기회** — 문헌 공백을 메운다.
**리스크** — 심사자가 이 지점을 물으면 **프로젝트 자체 측정이 유일한 방어선**이다.
최소한 (DSP 기반 INT8 MAC) vs (LUT 기반 bit-serial) 을 **동일 FPGA 에서
LUT · FF · BRAM · Fmax · BRAM read 수로 직접 비교한 표**가 필요하다.

### 4-1. work-compaction 논거 — 외부 근거는 Salca 한 건뿐

Salca 본문 (abstract 아님) 에 이런 서술이 있다.

> *"sparse attention degrades actual memory access efficiency in two aspects.
> First, **discrete accesses and short burst transfers destroy spatial locality**
> for memory reads. Second, parallel index-based access causes physical conflicts
> across memory channels. The two factors reduce **HBM transfer efficiency from
> 95% to 30%**. This throttles data supply and leaves computing units starving."*

대응책도 compaction 인접이다 — *"we employ a single-PC mapping strategy rather
than across-PCs for individual K/V distribution ... This maximizes burst length."*
추가로 stride=1 maxpooling 으로 *"positions surrounding high-score elements are
co-selected, therefore enhancing locality"* — 선택 단계에서 연속성을 복원한다.

**한정이 중요하다.** Salca 의 근거는 **off-chip HBM burst / channel 충돌** 층위이고,
프로젝트의 발견은 **on-chip BRAM word-width** 층위다.

**→ CITE AS CROSS-LEVEL CORROBORATION.** "메모리는 종료 여부와 무관하게 고정된
넓은 granule 단위로 읽는다"는 **원리**의 외부 근거로만 인용하고,
**BRAM 층위의 증명으로 제시하면 과장이다.**

---

## 5. Angle 5 — 반증과 타당성 위협

### 5-1. ★ vAttention — 가장 강한 정확도 반증 (정면 대응 필요)

> Desai, Agrawal, Yang, Cuadron, Schroeder, Zaharia, Gonzalez, Stoica.
> **"vAttention: Verified Sparse Attention."** ICLR 2026.
> arXiv:[2510.05688](https://arxiv.org/abs/2510.05688)

**exact / oracle top-k 조차 full attention 을 항상 근사하지 못한다**는 것을 보인다.

* Abstract: *"top-k and random sampling are complementary: top-k performs well
  when attention scores are dominated by a few tokens, whereas **random sampling
  provides better estimates when attention scores are relatively uniform**."*
* 본문은 더 강하다: *"**even access to the exact top-k tokens** under full attention
  **does not always suffice** to approximate the original output"*
* *"the ordering is **inconsistent** — for a given query across heads, or for a
  given head across queries"* → **단일 전역 θ 상수는 방어할 수 없다**
* Table 1: oracle-top-k 88.61% vs full attention 88.74% (RULER-HARD)

**→ THREAT TO VALIDITY, 정면 대응 필수.**
프로젝트의 branch-and-bound 는 정확한 top-k 를 고르는 selector 인데,
이 논문은 "정확한 top-k 라도 충분하지 않을 때가 있다"를 보인다.
**"정확한 bound 니까 무손실"이라는 논증 경로가 막힌다.**
최소한 head/query 별 score 분포 편차 측정 + perplexity / long-context 벤치마크가 필요하고,
**θ 는 head 별로 달라야 한다.**

> 📌 **정정**: "attention-entropy check" 라는 처방은 이 논문의 제안이 **아니다**.
> 'entropy' 는 본문에 단 한 번도 등장하지 않는다. 귀속하면 안 된다.

> **범위 한정**: vAttention 은 post-softmax weight 와 출력을 분석하고,
> 프로젝트는 pre-softmax Q·K^T score 를 임계화한다. 또 vAttention 이 논하는 것은
> token 예산 k 이고 프로젝트의 θ 는 score 임계값이다.

### 5-2. Twilight — 대비용으로 유용

> Lin, Tang, Yang, Wang, Tang, Tian, Stoica, Han, Gao. **"Twilight: Adaptive
> Attention Sparsity with Hierarchical Top-p Pruning."** NeurIPS 2025 (spotlight).
> arXiv:[2502.02770](https://arxiv.org/abs/2502.02770)

top-p 임계화가 출력 오차에 **(1−p)·||V||_F 라는 이론적 상한**을 준다.
*"Compared to top-k, top-p is more advantageous because it provides a theoretical
upper bound of error as (1−p)·||V||_F."* 단 **정확성을 주장하지 않는다** —
*"nearly no accuracy loss"*, Sec.5.1 에서 *"<1%"*. 구성상 (1−p) 만큼의
attention mass 를 버리므로 p<1 이면 오차 하한이 0이 아니다.

**→ CITE AS CONTRAST.** 기여 프레이밍에 유용하다.

| | Twilight | 이 프로젝트 |
|---|---|---|
| 대상 | 출력 오차 | **score 자체** |
| 성격 | norm bound, 근사 | **결정론적 정확 구간** |
| 시점 | 사후 | **버리기 전에 확정** |

단 프로젝트의 end-to-end 무손실성은 bracket 이 아니라 **θ 선택 정책**에 달려 있다.
**"bracket 은 정확하다"와 "시스템이 무손실이다"를 구분해 서술할 것.**

### 5-3. 확보하지 못한 반증

"decode 에서 attention 은 memory-bandwidth-bound 라 연산 절감이 end-to-end 로
전달되지 않는다"는 **시스템 측 반증을 확보하지 못했다.** 관련 주장 2건이 모두 검증에서
기각되었다(§7). 이것은 프로젝트 논거의 **핵심 전제**(architecture.md §0) 이므로,
없다는 것이 유리하지만은 않다 — 심사자가 반대 방향으로 물을 수 있다.

---

## 6. 정독 우선순위

| # | 논문 | 왜 |
|---|---|---|
| **1** | **PADE** (arXiv:2512.14322) | 메커니즘이 거의 동일. 기여 재정의와 차별화 문장을 여기 맞춰 다시 써야 한다. BUI 양방향 구간 vs 단측 R_m, BS-OOE vs work-compaction 의 차이를 정확히 파악할 것 |
| **2** | **BitStopper** (arXiv:2512.06457) | 이미 [4]로 인용 중인데 수치 귀속이 틀렸다. Sec. III-B 의 2's complement bound 유도를 읽고 unsigned 전제 문구를 어떻게 고칠지 결정 |
| **3** | **BETA** (TCAS-II 2025) | SpAtten[2]을 6.5x 로 추월한 peer-reviewed 후속. **MTS 수식이 paywall 뒤에 있어 미확인** — θ 설계 근거를 굳히려면 본문 확보 필요 |
| **4** | **vAttention** (arXiv:2510.05688) | 유일하게 살아남은 강한 정확도 반증. 무손실성 주장 전체를 좌우한다 |
| **5** | **Salca** (arXiv:2604.24820) | work-compaction 논거의 **사실상 유일한 외부 근거**(HBM 95%→30%) + 히스토그램 O(n) top-k |

차순위: **MCBP** (arXiv:2509.10372) — radius 기반 θ 갱신 모듈 원형.
**Titanus** (arXiv:2505.17787) — Angle 3 각주용.

---

## 7. 기각된 주장 — 쓰면 안 되는 것 (투명성)

3표 적대적 검증에서 **11건이 기각**되었다. 그럴듯해 보였지만 원문 대조에서 무너진 것들이다.

| 기각된 주장 | 표 |
|---|---|
| BETA 의 주된 성과가 cycle 이 아니라 memory access 절감이다 | 0-3 |
| BETA 의 BOOE 가 work-compaction 필요성의 외부 근거다 | 0-3 |
| PADE 의 BS-OOE 가 같은 근거다 | 0-3 |
| BitStopper 가 Sanger/SOFA 의 "연산은 줄었으나 메모리는 안 줄었다"를 보고한다 | 1-2 |
| Salca 가 K 를 2-bit asymmetric 으로 양자화한다 | 0-3 |
| Twilight 의 p 가 오프라인 캘리브레이션으로 정해진다 | 0-3 |
| Twilight / GVR 이 decode 의 memory-bandwidth 병목을 정량 입증한다 | 1-2 |
| GVR 의 secant-style counting 임계값 | 0-3 |
| **이전 step top-k 재사용의 verify 기반 무손실성** | **1-2** |

**특히 마지막 항목** — "verify 단계를 붙이면 이전 step θ 재사용이 무손실"이라는 주장이
기각되었다는 것은, 그 무손실성에 **검증된 답이 없다**는 뜻이다.

**중요:** 앞 4건이 기각되었다는 것은 **"외부 문헌이 프로젝트의 BRAM 발견을 확증한다"고
쓸 수 없다**는 뜻이다. 관찰 사실 자체는 남아 있다 — PADE 는 BS-OOE 를, BETA 는 BOOE 를
각각 별도 기여로 두고 있고 이는 흩어진 bit-level 작업을 재정렬/압축하는 기구다.
그러나 검증자들은 이 논문들이 그 기구의 **동기**를 "메모리 접근 granule 때문에 sparsity 가
실현되지 않는다"로 서술한다고 볼 수 없다고 판단했다 (주된 서술은 PE 부하 불균형 / 이용률 회복).

**→ "PADE/BETA 가 OOE 기구를 필요로 했다는 점은 정황적으로 시사적" 정도로만 약하게 쓸 것.**

---

## 8. 열린 질문

1. **`prev_step` θ 재사용은 무손실인가?** 문헌에 답이 없다. 프로젝트가 자체 반례를 갖고 있으므로
   작은 기여가 될 수 있다.
2. **★ QuaRot / SpinQuant 등 rotation·Hadamard 기반 양자화가 bit-plane 단조성을 깨는가?**
   이번 조사에서 **전혀 검증하지 못했다.** 회전 후 값이 부호 있는 준가우시안이 되므로
   **실제 위험 가능성이 있다.** Angle 3 의 최대 미해결 항목.
3. **최신 Xilinx 부품에서 LUT 기반 bit-serial 이 iso-area 로 DSP 기반 INT8 MAC 을 이기는가?**
   published 데이터를 찾지 못했다. 직접 비교표를 만들어야 한다.
4. **on-chip BRAM word-width 대 종료 실현 가능성을 정량 보고한 문헌이 존재하는가?**
   확보한 것은 Salca 의 off-chip 근거뿐. 고유 기여일 수도, 탐색 실패일 수도 있다.
5. **decode attention 이 memory-bandwidth-bound 라는 정량 근거는 어디에 있는가?**
   관련 주장 2건이 기각되어 미해결.

---

## 9. 조사 한계

* **검증 절차**: 일부 세션에서 WebSearch 예산(200/200)이 소진되어 PADE · Salca · Titanus 에 대한
  **제3자 반박 문헌 스윕을 완료하지 못했다.** primary source 원문 대조로는 확정적이지만,
  "아무도 반박하지 않았다"는 부분은 검증되지 않았다.
* **문헌 등급 차이**: BitStopper 와 Salca 는 **미심사 preprint**. PADE 의 HPCA 2026 채택은
  공식 프로그램 + IEEE Xplore 로 독립 확인, BETA 는 Crossref/OpenAlex 확인된 peer-reviewed
  IEEE 저널, Twilight 은 NeurIPS 2025 spotlight, vAttention 은 ICLR 2026.
* **미확인 항목**: (a) BETA 의 MTS 수식 본문 — paywall. 'running max 기반'이라는 성격은
  abstract + 동일 그룹 논문에서 **추론**한 것. (b) BETA 에 FPGA 데이터가 없다는 점도 **추론**.
  (c) BitStopper 오버헤드 합계는 **단순 합**으로 엄밀한 가산성 없음.
* **각도별 불균형**: Angle 1·2 는 primary source 로 충실히 답했다.
  **Angle 3 은 Titanus 한 건, Angle 4 는 사실상 공백**, Angle 5 는 정확도 측면만 커버.

---

## 10. 조사 통계

| | |
|---|---|
| 조사 각도 | 5 |
| fetch 한 primary source | 22 |
| 추출된 검증 가능 주장 | 110 |
| 3표 적대적 검증 대상 | 25 |
| **확정** | **14** |
| **기각** | **11** |
| 실행 agent | 104 |
| 조사 시각 | 2026-08-16 |
