# 파일 구조 설명

LLM 어텐션 가속기 — **상위 비트 우선 계산 + 조기 종단**의 소프트웨어 검증 계층.
배경지식 가이드의 8.3절 "1단계 — 알고리즘 수준 검증"에 해당하며,
2~4단계(RTL 구현·합성·측정)와는 `config/hardware.yaml` 과 `rtl_data/` 로만 연결된다.

---

## 0. 설계 원칙 네 가지

> 설계 자체(무엇을 왜 이렇게 만드는가)는 **[architecture.md](architecture.md)** 를 보면 된다.
> 이 문서는 그 설계가 **파일로 어떻게 배치되어 있는지**를 다룬다.

이 구조가 왜 이렇게 생겼는지는 아래 네 원칙으로 전부 설명된다.

### ① 팀 간 의존성을 파일 두 개로 끊는다

RTL 이 완성되기 전에도 알고리즘 검증이 끝까지 진행되어야 한다.

```
        [RTL / FPGA 팀]                    [소프트웨어 팀]
              │                                   │
              │  config/hardware.yaml  ──────────►│   (파라미터: 워드폭, 지연, Fmax…)
              │  rtl_data/*.csv, *.rpt ──────────►│   (실측: 사이클, 읽기, 자원)
              │                                   │
              └───────  utils/crosscheck.py  ◄────┘   (예측 vs 실측 대조)
```

RTL 이 없으면 `rtl_data/mock/` 으로 파이프라인 전체가 돈다.
실측이 들어오면 같은 이름으로 최상위에 놓기만 하면 된다.

### ② 실험과 그림을 분리한다

```
experiments/*.py  ──►  outputs/raw/*.csv  ──►  utils/visualization.py  ──►  outputs/figures/
```

**그림 생성은 `outputs/raw/` 만 읽는다.** 따라서 실험을 다시 돌리지 않고도
`--figures-only` 로 figure 를 언제든 재생성할 수 있다.

### ③ 부분 내적을 한 번만 계산한다

`P_b` 는 margin / θ정책 / top-k / 스케줄 정책과 **무관**하다.
`DecodeWorkbench` 가 한 번 계산해 두고 모든 스윕이 재사용한다.
덕분에 seq_len=512 기준 스윕 하나가 수 초 안에 끝난다.

### ④ 이론값과 실현값을 항상 함께 낸다

절감을 한 숫자로만 보고하면 발표에서 무너진다. 모든 절감 지표는 쌍으로 나온다.

| 지표 | 의미 |
|---|---|
| `read_saving_ideal` | 살아있는 (토큰, 평면) 쌍 기준 — **이론** |
| `read_saving_bram` | 실제 BRAM 워드 읽기 기준 — **실현** |
| `read_realization_ratio` | 실현/이론. 1.0 이면 완전 실현 |

---

## 1. 전체 트리

```
llm_et_attention/
├── run_paper_experiments.py     오케스트레이터 (--only / --skip / --figures-only / --crosscheck)
├── README.md                    재현 가이드 — ★ 루트에 두는 .md 는 이것 하나뿐이다
│
├── docs/                        ★ 문서는 전부 여기
│   ├── README.md                ★ 문서 진입점 (목차 + 무엇을 언제 읽나)
│   ├── architecture.md          전체 조감도 해설 (설계 자체)
│   ├── structure.md             이 문서
│   ├── related_work.md          선행연구 조사 (2026-08)
│   └── background/              LLM 어텐션 기초
│       ├── transformer.md            Transformer · Self-Attention · Prefill/Decode
│       ├── attention_walkthrough.md  ★ 손으로 따라가는 수치 예제 + 종단 판정
│       └── llama_3_2_1b.md           대상 모델 스펙·성능, KV 캐시 크기 계산
│
├── slides/                      ★ 그림은 "쓰는 문서" 기준으로 나눈다
│   ├── README/                  README.md 용
│   │   └── 00_overview.svg
│   ├── architecture/            docs/architecture.md 용 (폴더 이름은 그대로 둔다)
│   │   ├── 00_overview.svg
│   │   ├── 05_finding.svg
│   │   └── bitplane_layout.svg  §1-4 비트평면 저장 방식
│   ├── docs_background/         docs/background/*.md 용
│   └── deck/                    발표용 16:9 슬라이드 6장
│       └── 00_overview ~ 05_finding.svg
│
│   ※ 같은 그림이 여러 문서에 쓰이면 폴더마다 사본을 둔다.
│     문서별로 따로 손볼 수 있게 하려는 의도이며, 사본끼리 자동 동기화되지 않는다.
│
├── config/                      ★ 설정만 고치면 실험 범위가 바뀐다 (코드 무수정)
│   ├── model.yaml               Llama 3.2 1B 스펙 + 디코드 루프 설정
│   ├── quant.yaml               ★ 양자화 규약 (상한식 성립의 전제)
│   ├── hardware.yaml            ★ RTL 팀 인터페이스 (워드폭·지연·자원·Fmax)
│   └── sweeps.yaml              exp1~exp6 의 스윕 범위
│
├── src/                         알고리즘 코어 (numpy 만 필요)
│   ├── config.py                YAML 로더 (PyYAML 없으면 내장 미니 파서)
│   ├── seeding.py               난수 시드 관리
│   ├── quantize.py              ★ 비대칭 unsigned K + zero-point / 대칭 signed q
│   ├── bounds.py                ★ Q+/Q−, R_m, L_m
│   ├── masked_sum.py            P_b (곱셈기 없는 조건부 덧셈) + 가산 트리 모델
│   ├── accumulator.py           S_m 시프트 누산 + zero-point 보정
│   ├── threshold.py             θ 관리부 (정책 4종)
│   ├── terminator.py            조기 종단 판정 (판정 지연 포함)
│   ├── designs.py               ★ ①②③④ 를 동일 인터페이스로
│   ├── schedule.py              종단 불규칙성 처리 (batch/compaction/two_phase)
│   ├── memory.py                ★ 비트평면 BRAM 모델 + 읽기 회계
│   ├── decode_loop.py           ★ 디코드 루프 시뮬레이터 (T 가 자란다)
│   ├── dataset.py               합성 Q/K (attention sink 포함) + 캐시 로드
│   └── model_hooks.py           실제 Llama q/K 캡처 → cache/tensors/
│
├── utils/                       분석 · 파싱 · 시각화
│   ├── metrics.py               top-k 보존율, 종단 프로파일, 절감 분해
│   ├── cost_model.py            ★ 자원·Fmax 감안 손익분기
│   ├── hw_parser.py             Vivado rpt / sim csv 파싱
│   ├── crosscheck.py            ★ SW 예측 vs RTL 실측 대조
│   ├── io.py                    raw 저장/로드 + 재현성 스탬핑 + LaTeX 표
│   ├── visualization.py         논문용 그래프 (검증된 팔레트)
│   └── visualization_example/   웹 교보재 — 코어는 이쪽을 임포트하지 않는다
│       ├── matmul.py            /            행렬 곱 + numpy 교차 검증
│       ├── schedule_demo.py     /schedule    src.schedule.apply() 를 그대로 구동
│       ├── anatomy.py           /schedule_py inspect 로 코드에서 읽는 해부도
│       ├── glossary.py          /glossary    용어 — 한 격자 위에 겹쳐 본다
│       ├── server.py            http.server 기반 (추가 의존성 없음)
│       └── static/              화면 — 산술은 하지 않고 그리기만 한다
│
├── experiments/                 실험 6종 (각각 단독 실행 가능)
│   ├── exp1_termination_profile.py   ★ 가장 먼저 — 종단이 일어나는가
│   ├── exp2_margin_sweep.py          ★ 최종 산출물 — 그림 8.1
│   ├── exp3_theta_policy.py          θ 확정 시점 (6.3-3)
│   ├── exp4_schedule_policy.py       ★ BRAM 워드폭 함정 (6.3-2)
│   ├── exp5_seqlen_topk.py           문맥 길이 / 상위 k 스캔
│   └── exp6_breakeven.py             ★ 손익분기 (6.3-4)
│
├── rtl_data/
│   ├── schema.md                ★ RTL 팀과의 데이터 계약
│   └── mock/                    RTL 없이도 파이프라인이 돌게 하는 더미
│
├── tests/                       209개 (pytest 없이도 실행 가능)
│   ├── run_tests.py             내장 러너
│   ├── test_schedule.py         정책 + ★ 설정 배선 (값 비교로는 못 잡는다)
│   ├── test_memory_cycles.py    ★ BRAM 포트 -> 메모리 사이클 -> 병목 판정
│   └── test_score_budget.py     ★ 정책을 바꿔도 점수가 문턱 안인가
│
├── cache/tensors/               캡처한 q/K (gitignore)
└── outputs/{raw,figures,tables,logs}   전부 재생성 가능 (gitignore)
```

---

## 2. 수학 규약 — ★ 함부로 바꾸면 안 되는 부분 ★

가이드 5.3 / 5.4절에 대응한다. `config/quant.yaml` 이 이 규약을 고정한다.

### K 는 unsigned 로 저장한다

```
K_stored ∈ [0, 255],   실제값 ≈ (K_stored − z) · scale_k
비트 자리값 = [128, 64, 32, 16, 8, 4, 2, 1]   ← 전부 양수
```

이 성질 때문에 "남은 비트가 전부 1"이 곧 최대 기여가 되고,

```
R_m = (2^(8−m) − 1) · Q+          Q+ = q 원소 중 양수의 합
L_m = S_m + (2^(8−m) − 1) · Q−    Q− = q 원소 중 음수의 합
```

이 **정확한** 상하한이 되고, 8개 평면 전부에 이 단일 공식이 그대로 적용된다.
`tests/test_quantize.py::test_plane_weights_all_positive` 가 이를 지킨다.

> **★ 정정 (2026-08)** — "signed 로 저장하면 상한식이 깨진다"는 이전 서술은 **틀렸다.**
> MSB-first 라 부호 비트는 라운드 0에 확정되어 결정된 부분합에 들어가고,
> 미확정 비트의 자리값은 전부 양수다. signed 에서도 유효한 bound 를 세울 수 있으며
> PADE(HPCA 2026)·BitStopper 가 그렇게 한다. unsigned 는 **평면 0의 부호 특수처리와
> 양방향 구간 로직을 없애는 회로 단순화**이지 수학적 필연이 아니다.
> 자세한 것은 [related_work.md](related_work.md).

### per-channel 양자화와 "곱셈기 없음"을 동시에 지키는 법

K 를 차원별 scale 로 양자화하면(KIVI 권장) `sc_i` 가 내적 안쪽에 남아 곱셈이 되살아난다.
해결책은 **sc_i 를 q 에 미리 접어 넣는 것**이다 (`accumulator.fold_and_quantize_query`).

```
q̃_i = q_real_i · sc_i         (소프트웨어에서 스텝당 d회 — 하드웨어 밖)
s_real = scale_q · ( Σ q_st_i·K_st[j,i]  −  Σ q_st_i·z_i ) / √d
                     └─ 마스크드 합 (곱셈 없음) ─┘  └ 스텝당 상수 ┘
```

두 번째 항은 **모든 토큰에 동일한 상수**이므로
* 순위(top-k) 판정에는 영향이 없다 → 종단 로직은 정수 `s_int` 만 본다
* 최종 점수값에는 필요하다 → `accumulator.to_real_scores` 가 더한다

---

## 3. 데이터 흐름

```
   snapshot (q, k)                 dataset.py / model_hooks.py
        │
        ▼  quantize_key + fold_and_quantize_query      quantize.py / accumulator.py
   K_stored (uint8), q_st (int8), zero-point 보정항
        │
        ▼  to_bitplanes → partial_dots                 quantize.py / masked_sum.py
   P[b, s, j]   (8, n_steps, n_tokens)     ★ 한 번만 계산, 모든 스윕이 재사용 ★
        │
        ▼  디코드 루프 (스텝마다 T 가 자람)              decode_loop.py
        │
        ├─► bounds.step_bounds(q_st)  →  Q+, Q−         bounds.py
        ├─► terminator.run_step        →  live 마스크    terminator.py + threshold.py
        ├─► schedule.apply             →  사이클         schedule.py
        ├─► memory.account_step        →  BRAM 읽기      memory.py
        └─► metrics.accuracy_metrics   →  top-k 보존율   utils/metrics.py
                     │
                     ▼
              outputs/raw/*.csv  ──►  figures / tables
```

---

## 4. 네 비교 설계 (가이드 8.1절)

`src/designs.py` 가 넷을 **같은 시그니처**로 노출한다.

| # | 이름 | 설명 | 목적 |
|---|---|---|---|
| ① | `baseline` | 병렬 INT8 곱셈 누산 | 기준선. DSP 사용 |
| ② | `seq` | 비트평면 순차, 종단 없음 | 순차 전환 비용 분리 |
| ③ | `exact` | + 정확 모드 종단 | 무손실 순수 이득 |
| ④ | `approx` | + 근사 모드 종단 | 절감–정확도 곡선 |

**②와 ③의 비교가 승부처다.** ①과의 비교는 사이클에서 구조적으로 불리하다
(비트평면은 8사이클을 쓴다). 제안의 근거는 사이클이 아니라
**(a) DSP 미사용 (b) 메모리 읽기 감소**이며, `exp6` 가 이 사실을 숨기지 않는다.

불변식 — `tests/test_designs.py` 가 검증한다:
* ① = ② = ③ 의 점수는 **비트 단위로 동일**
* ③ 은 참 top-k 를 절대 잃지 않는다
* margin=0 인 ④ 는 ③ 과 완전히 같다

---

## 5. 함정 세 가지와 그 대응

이 구조가 방어하는, 반박당하기 쉬운 지점들이다.

### 함정 1 — BRAM 워드 단위 읽기 (`src/memory.py`)

가이드 5.7절은 "메모리 읽기 절감이 더 중요"라고 했다. 맞다. 그런데:

```
한 워드에 word_tokens 개 토큰의 같은 평면 비트가 묶여 있다
  → 워드 안에 살아있는 토큰이 하나라도 있으면 워드 전체를 읽어야 한다
  → 종단된 토큰이 흩어져 있으면 절감이 실현되지 않는다
```

실측 결과 (word_tokens=32, 정확 모드): 이론 절감 27.9% 가

* `batch` 정책 → 실현 **3.8%** (실현률 14%)
* `compaction` 정책 → 실현 **26.6%** (실현률 95%)

이것이 work-compaction 이 필요한 진짜 이유이며, `exp4` 가 정량화한다.

### 함정 2 — 판정 지연 (`config/hardware.yaml: decision_latency_planes`)

평면 m 의 종단 판정은 평면 m 의 가산 트리가 끝나야 나온다. 그 시점에 평면 m+1 의
읽기 요청은 이미 나갔을 수 있다. 이 값을 0으로 두면 읽기 절감이 과대평가된다.
`tests/test_designs.py::test_decision_latency_reduces_savings` 가 이 장치가
살아 있는지 확인한다.

### 함정 3 — `prev_step` θ 정책은 무손실이 아니다 (`src/threshold.py`)

θ 를 직전 디코드 스텝에서 가져오면 부분 정렬 회로가 필요 없어 회로가 단순해진다.
그런데 θ 가 이번 스텝의 하한에서 나온 것이 아니므로 **참 top-k 가 잘릴 수 있다.**
margin=0 인데도 손실이 나는 유일한 정책이며, `exp3` 가 이 교환을 정량화한다.
(실측: oracle 의 90% 수준 절감을 얻지만 무손실 보장은 사라진다)

---

## 6. 재현성 장치

| 장치 | 위치 |
|---|---|
| 설정 해시 + git commit + 타임스탬프 스탬핑 | `utils/io.py: provenance()` → `outputs/raw/*.meta.json` |
| `source: estimate` 파라미터 자동 경고 | `src/config.py: provenance_warnings()` |
| 시드 분리 (데이터 / 시행) | `src/seeding.py` |
| PyYAML 없이도 config 파싱 | `src/config.py: _mini_yaml_load()` |
| pytest 없이도 테스트 실행 | `tests/run_tests.py` |

`cfg.provenance_warnings()` 가 빈 목록이 되면 (= 모든 하드웨어 값이 실측으로 교체되면)
논문에 쓸 준비가 된 것이다.

---

## 7. 알려진 한계 (논문에 명시할 것)

1. **합성 데이터** — torch/transformers 없이 돌 때는 `src/dataset.py` 의 합성 Q/K 를 쓴다.
   attention sink 를 심고 로짓 표준편차를 실제 수준(≈5)으로 맞췄지만,
   논문 본문 수치는 `model_hooks.py` 로 캡처한 실제 텐서로 다시 뽑아야 한다.
2. **RoPE 미적용** — `model_hooks.py` 는 q_proj/k_proj 출력을 잡는다.
   회전이므로 내적 분포의 스케일은 크게 바뀌지 않으나 상대 위치 효과는 빠져 있다.
3. **단일 층·단일 헤드** — 가이드 1.3절의 범위 그대로. 나머지는 해석적 추정.
4. **하드웨어 값은 추정치** — `config/hardware.yaml` 의 `source` 가 전부
   `estimate` 인 상태에서는 exp6 의 손익분기 결론이 잠정적이다.
