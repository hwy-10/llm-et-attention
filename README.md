# LLM 어텐션 가속기 — MSB-first 조기 종단 (FPGA RTL 프로젝트)

[![tests](https://github.com/hwy-10/llm-et-attention/actions/workflows/tests.yml/badge.svg)](https://github.com/hwy-10/llm-et-attention/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**최종 산출물은 FPGA 에 올릴 RTL 이다.** 상위 비트 우선(MSB-first) 비트평면 계산 +
조기 종단(early termination)을 하드웨어로 구현하고, **이 기법이 실제로 이득이 되는
조건의 경계**를 RTL 실측까지 포함해 정량화한다.

![프로젝트 전체 구성](slides/00_overview.svg)

이 저장소의 파이썬 스택은 결과물이 아니라 **RTL 의 골든 레퍼런스**다.

* **정답 기준** — 사이클 단위로 동작이 확정된 알고리즘 모델. RTL 출력이 맞는지 판정한다.
* **설계 공간 탐색** — 워드폭·분할점 m0·margin·θ 정책을 RTL 을 짓기 *전에* 확정한다.
  잘못 고르면 Verilog 를 다시 쓰는 비용이 든다.
* **실측 대조** — Vivado 시뮬레이션·합성 결과를 받아 예측과 어긋나면 잡아낸다
  (`--crosscheck`). 어긋나면 소프트웨어 모델이 틀린 것이다.

즉 검증 범위는 알고리즘에서 끝나지 않고 **RTL 실측 대조까지** 이어진다.
소프트웨어만으로 "이득이 있다"고 말하지 않는다 — 실제로 지금도 추정 Fmax 기준으로는
손익분기 미달이며(아래 exp6), 그 결론은 Vivado 실측이 들어와야 확정된다.

배경지식 가이드 8.3절 기준으로 이 저장소는 **1단계(알고리즘 검증)** 를 맡고,
2~4단계(RTL·합성·측정)와는 `config/hardware.yaml` 과 `rtl_data/` **두 지점으로만**
맞물린다. 덕분에 RTL 이 완성되기 전에도 실험이 끝까지 진행되고,
RTL 이 나오면 같은 저장소에서 곧바로 대조된다.

> **[architecture.md](architecture.md)** — 설계 전체 조감도 해설 (여기부터 읽으면 된다)
> **[STRUCTURE.md](STRUCTURE.md)** — 파일 구조와 설계 원칙
> **[slides/](slides/)** — 발표용 16:9 슬라이드 6장

---

## 빠른 시작

의존성은 **numpy 하나**다. 나머지는 전부 선택.

```bash
pip install numpy                      # 필수
pip install matplotlib                 # 그림 (없으면 CSV 만 나옴)
pip install PyYAML pandas pytest       # 선택 (없어도 내장 대체 구현이 동작)
```

```bash
python tests/run_tests.py              # 55개 테스트 — 먼저 이걸 통과시킬 것
python run_paper_experiments.py        # 전체 실험 + 그림 + 표  (약 1분)
```

```
outputs/raw/      실험 원본 레코드 (CSV) + 재현성 메타 (JSON)
outputs/figures/  논문용 벡터 그림 (PDF + PNG)
outputs/tables/   LaTeX / CSV 표
```

### 자주 쓰는 실행

```bash
python run_paper_experiments.py --only exp1          # 종단이 일어나는지 먼저 확인
python run_paper_experiments.py --only exp2 exp4     # 일부만
python run_paper_experiments.py --figures-only       # 실험 없이 그림만 재생성
python run_paper_experiments.py --crosscheck         # RTL 실측과 대조
python run_paper_experiments.py --generate-mock      # mock 실측 CSV 재생성
python run_paper_experiments.py --quick              # 짧은 시퀀스로 빠르게 점검

python -m experiments.exp2_margin_sweep              # 실험 단독 실행
```

---

## 실험 여섯 개

| 실험 | 답하는 질문 | 가이드 |
|---|---|---|
| **exp1** 종단 프로파일 | 평균적으로 몇 번째 평면에서 종단되는가? | 8.3 1단계 |
| **exp2** margin 스윕 | 여유값 대비 절감량과 정확도 손실 — **최종 산출물** | 8.4 / 그림 8.1 |
| **exp3** θ 정책 | θ 를 언제 확정할 것인가 | 6.3-(3) |
| **exp4** 스케줄 정책 | 종단 불규칙성 처리 + **BRAM 워드폭 함정** | 6.3-(2), 5.7 |
| **exp5** N × k 스캔 | 문맥 길이와 상위 k 에 따라 결론이 어떻게 바뀌는가 | 6.3-(4) |
| **exp6** 손익분기 | 자원·Fmax 저하를 감안하고도 이득이 남는가 | 6.3-(4), 7.3 |

**exp1 을 가장 먼저 돌린다.** 종단이 거의 일어나지 않으면 이후 RTL 구현이
전부 무의미하므로, 그 시점에 설계를 재검토해야 한다.
exp1 은 정확 모드의 무손실성도 함께 자동 점검한다.

---

## 합성 데이터 기준 예시 결과

`seq_len=512`, `head_dim=64`, `word_tokens=32`, `decision_latency=1`, 정확 모드 기준.
**아래는 합성 Q/K 기준이며 경향 확인용이다. 논문 수치는 실제 캡처로 다시 뽑아야 한다.**

### 평면별 생존 곡선 — 종단은 평면 4부터 시작된다

```
평면(MSB→LSB)   0     1     2     3     4     5     6     7
생존 비율      100%  100%  100%  100%   95%   56%   18%    7%
```

→ 2단계 처리(6.3-(2))의 분할점 m0 = 4 가 자연스러운 선택이 된다.

### 그림 8.1 — 여유값 대비 절감량 / 정확도 (top-8)

| margin | 읽기 절감 (실현) | 읽기 절감 (이론) | top-8 보존율 |
|---|---|---|---|
| 0.00 (정확 모드) | 26.6% | 27.9% | **1.0000** |
| 0.70 | 45.6% | 48.5% | **1.0000** |
| 0.80 | 53.4% | 56.9% | 0.9997 |
| 0.90 | 63.2% | 67.8% | 0.9503 |
| 1.00 | 66.6% | 71.5% | 0.4339 ← 붕괴 |

### ★ BRAM 워드폭 함정 — 이론과 실현의 격차

이론 절감 27.9% 가 워드폭과 스케줄 정책에 따라 이렇게 갈린다.

| 워드폭 | `batch` | `compaction` |
|---|---|---|
| 1 | 27.9% (100%) | 27.9% (100%) |
| 8 | 12.3% (44%) | 27.6% (99%) |
| 32 | **3.8% (14%)** | **26.6% (95%)** |
| 64 | 1.5% (5%) | 24.4% (87%) |

→ **work-compaction 없이는 메모리 읽기 절감이 실현되지 않는다.**

### 정직하게 짚어야 할 것

* 비트평면 순차 처리는 8사이클을 쓰므로 **기준 설계(①)보다 사이클이 8배 많다.**
  (512 토큰 기준 4,305 → 34,440 사이클)
* 제안의 근거는 사이클이 아니라 **(a) DSP 미사용 (b) 메모리 읽기 감소** 다.
* 추정 Fmax 기준 exp6 의 실효 speedup 은 ② 대비 0.93 으로 손익분기 미달이다.
  Vivado 실측으로 교체하기 전까지 이 결론은 잠정적이다.
* 문맥이 길수록 유리하다: 사이클 절감이 N=128 에서 −41%, N=512 에서 +10%.

---

## 실제 모델 텐서 사용하기

기본값은 합성 Q/K 다. 실제 Llama 3.2 1B 텐서를 쓰려면:

```bash
pip install "torch>=2.0" "transformers>=4.40"
python -m src.model_hooks --seq-len 512 --layer 8 --head 0
```

`cache/tensors/` 에 덤프되고, 이후 모든 실험이 **자동으로** 이걸 우선 사용한다.

> 한 번 덤프해 두는 것이 필수다. exp2 만 해도 33회 디코드 루프인데,
> 매번 모델을 forward 하면 스윕이 불가능하다.

---

## RTL 팀과의 연결

```
① RTL 팀이 rtl_data/schema.md 형식으로 결과를 놓는다
     rtl_simulation_cycles.csv     (design, seq_len, top_k, margin, cycles, bram_reads)
     <design>_synth.rpt  또는  <design>.json

② config/hardware.yaml 의 resources.* 를 실측으로 갱신하고
   source 를 estimate → vivado_synth 로 바꾼다

③ python run_paper_experiments.py --crosscheck
     예측과 실측이 5% 안에 들어오는지 확인한다  (--tolerance 로 조정)
     어긋나면 소프트웨어 모델이 틀린 것 — 원인 후보를 함께 출력한다
```

실측이 없어도 `rtl_data/mock/` 으로 이 경로 전체가 동작한다.
mock 은 소프트웨어 모델에서 생성한 것이라 항상 통과하며, 배관 점검용이다.
설정을 바꾼 뒤에는 `--generate-mock` 으로 다시 만든다.

> ⚠ `bram_reads` 는 **BRAM 워드 수**이지 토큰 수가 아니다.
> 단위가 어긋나면 대조가 통째로 무의미해진다. `rtl_data/schema.md` 참조.

---

## 설정 바꾸기

코드를 건드리지 않고 `config/` 만 고친다.

| 파일 | 내용 |
|---|---|
| `model.yaml` | 모델 스펙, 디코드 루프 길이(`seq_len`, `warmup_tokens`), 합성 데이터 |
| `quant.yaml` | ★ 양자화 규약 — 상한식 성립의 전제이므로 함부로 바꾸지 말 것 |
| `hardware.yaml` | ★ RTL 팀 인터페이스 — 워드폭, 판정 지연, 자원, Fmax |
| `sweeps.yaml` | exp1~exp6 의 스윕 범위 |

`source: estimate` 인 항목은 실행 시마다 경고로 표시된다.
전부 실측으로 교체되어 경고가 사라지면 논문에 쓸 준비가 된 것이다.

---

## 테스트

```bash
python tests/run_tests.py            # pytest 없이도 동작
python tests/run_tests.py bounds     # 일부만
pytest -q                            # pytest 가 있으면 이쪽도 가능
```

핵심 테스트:

| 파일 | 검증 |
|---|---|
| `test_bounds.py` | ★ `L_m ≤ s ≤ S_m + R_m` 이 모든 평면·모든 토큰에서 성립 |
| `test_designs.py` | ★ ①=②=③ 비트 단위 일치, 정확 모드가 top-k 를 잃지 않음 |
| `test_memory.py` | ★ 워드폭이 넓으면 흩어진 절감이 실현되지 않음 |
| `test_quantize.py` | K 자리값이 전부 양수(상한식 전제), zero-point 보정 항등식 |
| `test_decode_loop.py` | 루프 전체에서 무손실성 유지, `prev_step` 은 손실 발생 |
| `test_schedule.py` | 스케줄은 회계만 바꾸고 점수는 불변 |

---

## 참고

배경지식 가이드 10장의 문헌 중 이 저장소가 직접 반영한 것:

* **[1] LeOPArd** (ISCA 2022) — 비트 단위 조기 종단 구조
* **[2] SpAtten** (HPCA 2021) — 상위 비트 우선 읽기, top-k 선택 하드웨어
* **[4] BitStopper** (2025) — 제어 회로 오버헤드 6.9% (`utils/cost_model.py` 비교 기준)
* **[11] KIVI** (2024) — K 는 채널 단위 비대칭 양자화 (`config/quant.yaml`)
