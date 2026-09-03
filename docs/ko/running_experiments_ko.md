# 실험 실행법 — 직접 돌리는 방법

**2026-08-29 기준.** 저장소 `llm-et-attention` (팀1+팀2 병합본)에서 돌립니다.

---

## 0. 준비 — 한 번만

### 0-1. 위치

```bash
cd <저장소 폴더>          # src/ config/ experiments/ 가 보이는 곳
```

현재 병합본은 여기 있습니다:

```
C:\Users\trump\AppData\Local\Temp\claude\...\scratchpad\t1
```

임시 폴더라 **날아갈 수 있습니다.** 아래 명령으로 안전한 곳에 복사해 두세요.

```bash
cp -r <위 경로> ~/Desktop/llm-et-attention
cd ~/Desktop/llm-et-attention
```

### 0-2. 필요한 것

| 무엇 | 없어도 되나 | 확인 |
|---|---|---|
| Python 3.10+ · NumPy | 필수 | `python -c "import numpy; print(numpy.__version__)"` |
| PyYAML | 없어도 됨 (자체 파서로 대체) | — |
| torch · transformers | **실제 텐서 실험에만** | `python -c "import torch, transformers"` |
| Llama 3.2 1B 가중치 | 실제 텐서 실험에만 | 아래 |
| wikitext-2 | 실제 텐서 실험에만 | 아래 |

**Llama와 wikitext는 이미 받아져 있습니다** (2.4 GB).

```bash
ls ~/.cache/huggingface/hub/
#   models--unsloth--Llama-3.2-1B
#   datasets--Salesforce--wikitext
```

없으면 처음 실행할 때 자동으로 받습니다. 이미 있으면 **오프라인으로 강제**하는 게 빠릅니다.

```bash
export HF_HUB_OFFLINE=1        # Windows PowerShell:  $env:HF_HUB_OFFLINE = "1"
```

### 0-3. 한글이 깨질 때

```bash
export PYTHONUTF8=1            # Windows PowerShell:  $env:PYTHONUTF8 = "1"
```

Windows 기본 코드페이지가 cp949라 이걸 안 주면 출력이 깨집니다. **항상 붙이세요.**

---

## 1. 제일 먼저 — 아무것도 안 깨졌는지

```bash
PYTHONUTF8=1 python tests/run_tests.py
```

```
통과 424  실패 0  건너뜀 0        <- 이게 나와야 한다
```

**실패가 하나라도 있으면 다른 걸 돌리지 마세요.** 수치를 믿을 수 없습니다.

pytest 없이 도는 자체 러너입니다. 특정 파일만 돌리려면:

```bash
PYTHONUTF8=1 python tests/run_tests.py --only test_terminator
```

---

## 2. 실험 전체 돌리기

```bash
PYTHONUTF8=1 python run_paper_experiments.py --no-figures
```

**T=2048 실제 텐서 기준으로 20~40분** 걸립니다. 그림까지 그리려면 `--no-figures`를 빼세요(matplotlib 필요).

### 하나만 돌리기

```bash
PYTHONUTF8=1 python run_paper_experiments.py --only exp2 --no-figures
PYTHONUTF8=1 python -m experiments.exp2_margin_sweep          # 이렇게도 된다
```

### 빠르게 훑기

```bash
PYTHONUTF8=1 python run_paper_experiments.py --quick --no-figures
```

`seq_len`을 128로 낮춰 돕니다. **논문 수치로 쓰면 안 됩니다** — 짧은 문맥은 판정 지연 때문에 절감이 크게 줄어듭니다(§8).

### 결과가 나오는 곳

```
outputs/raw/exp<N>_<이름>.csv          <- 숫자. 엑셀로 열면 된다
outputs/figures/                        <- 그림 (--no-figures 안 줬을 때)
```

---

## 3. 실험 9개 — 뭘 묻는 실험인가

| 실험 | 묻는 것 | 소유 | 시간 |
|---|---|---|---|
| `exp1` | 종단이 실제로 일어나는가, 어느 평면부터인가 | 팀 2 | 짧음 |
| `exp2` | margin을 얼마까지 밀 수 있나 | 팀 2 | 중간 |
| `exp3` | theta를 언제 확정할 것인가 | 팀 2 | 중간 |
| `exp4` | 스케줄 정책 × BRAM 워드폭 | 팀 1 | **김** |
| `exp5` | 문맥 길이와 상위 k 스캔 | 팀 1 | 중간 |
| `exp6` | Fmax 저하를 넘어서는 손익분기 (**연산축**) | 팀 1 | 짧음 |
| `exp7` | 병목 위치가 종단의 가치를 정한다 (**메모리 포함 축**) | 팀 2 | 중간 |
| `exp8` | **실측 perplexity** (보간 아님) | 팀 2 | **매우 김** |
| `exp9` | **margin 커버리지** — 전 헤드 × 여러 텍스트 | 팀 2 | **매우 김** |

> `exp6`과 `exp7`은 **같은 질문을 다른 축에서** 봅니다.
> `exp6`은 연산 사이클만 세므로 ① 대비로는 절대 안 이깁니다(비트평면이 그 축에서 8배 불리).
> `exp7`은 메모리를 포함해 같은 잣대로 재고, 거기서 T=2048이면 **이깁니다.**

---

## 4. 실제 텐서로 돌리기

기본은 합성 데이터입니다. 실제 Llama 텐서를 쓰려면 **먼저 캡처**해야 합니다.

### 4-1. 캡처

```bash
PYTHONUTF8=1 HF_HUB_OFFLINE=1 python -m src.model_hooks \
    --seq-len 2048 --layer 8 --head 0 --rope \
    --text-file cache/wikitext2_test.txt
```

```
[model_hooks] saved 2048 tokens (head_dim=64) -> cache/tensors/Llama-3.2-1B_L8_H0_T2048.npz
```

| 옵션 | 뜻 |
|---|---|
| `--seq-len` | 캡처할 토큰 수. `config/model.yaml`의 `seq_len`과 맞출 것 |
| `--layer` `--head` | 어느 층/헤드를 뽑을지 |
| `--rope` | **꼭 주세요.** RoPE 적용 후를 캡처합니다. 빼면 Q±폭이 12% 달라집니다 |
| `--text-file` | 없으면 짧은 기본 텍스트라 **2048이 안 채워집니다** |

`--text-file`에 줄 wikitext 파일이 없으면 이렇게 만듭니다.

```bash
PYTHONUTF8=1 HF_DATASETS_OFFLINE=1 python -c "
from datasets import load_dataset; import io
ds = load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test')
io.open('cache/wikitext2_test.txt','w',encoding='utf-8').write(
    '\n'.join(t for t in ds['text'] if t.strip()))
print('완료')"
```

### 4-2. 캡처했는지 확인

```bash
PYTHONUTF8=1 python -c "
from src.config import load_config
from src.dataset import snapshot_from_config
s = snapshot_from_config(load_config(), seq_len=2048)
print('출처:', s.source)
print('토큰:', s.n_tokens)"
```

```
출처: unsloth/Llama-3.2-1B:L8:H0:rope       <- 실제 텐서
출처: synthetic                              <- 합성. 캡처가 안 잡힌 것
```

**"synthetic"이 나오면** 캐시 파일명이 안 맞는 것입니다. 파일명은 이 규칙입니다.

```
cache/tensors/<모델명>_L<층>_H<헤드>_T<토큰수>.npz
```

`config/model.yaml`의 `layer_idx` / `head_idx` / `seq_len`과 일치해야 합니다.

---

## 5. 설정 바꾸기

숫자는 전부 `config/` 안에 있습니다. **코드를 고치지 마세요.**

| 파일 | 무엇 |
|---|---|
| `config/model.yaml` | 모델 스펙, `seq_len`(= N_MAX), 캡처할 층/헤드, 합성 데이터 파라미터 |
| `config/quant.yaml` | 평면 수, **`margin.value`**, 양자화 방식 |
| `config/hardware.yaml` | **RTL 팀과의 인터페이스.** lanes, word_tokens, n_bram_ports, 자원/Fmax 추정치 |
| `config/sweeps.yaml` | 실험별로 무엇을 훑을지 |

### 자주 바꾸는 것

```yaml
# config/model.yaml
decode:
  seq_len: 2048            # N_MAX. 늘리면 hardware.yaml 의 idx_bits 도!
  layer_idx: 8
  head_idx: 0

# config/quant.yaml
margin:
  value: 0.7               # 근사 모드 여유값

# config/hardware.yaml
memory:
  word_tokens: 1
  n_bram_ports: 32         # word_tokens x n_ports >= lanes 여야 한다
  decision_latency_mode: "auto"
output:
  out_buf: 2048            # = N_MAX
  idx_bits: 11             # = log2(N_MAX).  안 맞으면 예외로 막힌다
datapath:
  lanes: 32
```

> ### 함께 움직여야 하는 것 3쌍
>
> ```
> seq_len(N_MAX)  <->  out_buf,  idx_bits        어긋나면 예외
> word_tokens     <->  n_bram_ports              곱이 lanes 이상
> K_TOP           <->  out_buf                   out_buf >= K_TOP
> ```
>
> 테스트가 이걸 검사합니다. 하나만 고치면 `tests/run_tests.py`가 잡아 줍니다.

### 바꾼 뒤 반드시

```bash
PYTHONUTF8=1 python tests/run_tests.py                          # 424 통과 확인
PYTHONUTF8=1 python run_paper_experiments.py --generate-mock    # mock RTL 재생성
PYTHONUTF8=1 python run_paper_experiments.py --crosscheck       # 12/12 확인
```

**mock 재생성을 빼먹으면** crosscheck가 예전 설정과 비교해 불일치가 납니다.
실제로 `decision_latency_mode`를 `auto`로 바꿨을 때 이게 걸렸고, **그건 crosscheck가 제 일을 한 것**입니다.

---

## 6. 실험별 직접 실행

### exp8 — 실측 perplexity (가장 중요)

```bash
PYTHONUTF8=1 HF_HUB_OFFLINE=1 python -m experiments.exp8_real_perplexity \
    --tokens 1024 --top-k 16 --margin 0.0 0.7
```

| 옵션 | 뜻 |
|---|---|
| `--tokens` | 몇 토큰으로 perplexity를 잴지. 비용이 **제곱**으로 늡니다 — 아래 참고 |
| `--top-k` | 여러 개 줄 수 있습니다 |
| `--margin` | 여러 개 줄 수 있습니다 |
| `--per-layer` | 층을 하나씩 바꿔 봅니다(진단용). **기본은 전 층 동시 교체** |

> ### 규모를 먼저 계산하세요
>
> 비용이 **`tokens²  ×  top_k 개수 × (1 + margin 개수)`** 로 늡니다.
> `run_step`이 질의마다 활성 토큰 전체를 훑으므로 토큰 수에 **제곱**으로 붙습니다.
> `--tokens`를 2배로 하면 4배, 거기에 조합 수가 곱해집니다.
>
> **`save_records`는 맨 끝에 한 번만 불립니다.** `timeout`에 걸리면 **아무것도 안 남습니다.**
> 큰 격자를 돌릴 때는 `timeout`을 넉넉히 주거나 조합을 쪼개서 여러 번 돌리세요.

> **전 층을 동시에 바꾸는 게 기본입니다.**
> 층 8 하나만 바꾸면 나머지 15개 층이 full attention이라 열화가 거의 안 보입니다(+0.48%).
> 하드웨어는 **모든 층**을 대체하므로 그 조건으로 재야 논문에 쓸 수 있습니다.

**함께 나오는 세 값을 반드시 같이 보세요.**

```
full           손대지 않은 모델. 기준
oracle top-k   참 top-k 만 남긴 것. 우리가 이길 수 없는 하한
ours           우리 종단 로직
```

`ours < oracle`이면 **생존 집합이 상위집합이라는 성질이 품질로 확인**된 것입니다.

### exp9 — margin 커버리지

```bash
PYTHONUTF8=1 HF_HUB_OFFLINE=1 python -m experiments.exp9_margin_coverage \
    --heads 8 --layers 4 --seq-len 512
```

| 옵션 | 뜻 |
|---|---|
| `--heads` | 헤드 몇 개까지 (0번부터) |
| `--layers` | 층 몇 개를 균등 간격으로 |
| `--texts` | `wikitext` `code` `dialogue` 중 고르기 |

**조합 수 = 텍스트 × 층 × 헤드**입니다. `3 × 4 × 8 = 96`조합이면 오래 걸립니다.
처음엔 `--heads 2 --layers 2 --texts wikitext`로 감을 잡으세요.

이게 답하는 것:

```
전역 안전 margin    모든 조합에서 무손실인 최대값  <- 레지스터에 박을 값
헤드별 knee 분포    조합마다 얼마나 벌어지는가
헤드별 회로의 이득  이게 작으면 전역 하나로 간다 (면적 32배를 아낀다)
```

### exp7 — 병목 교차점

```bash
PYTHONUTF8=1 python -m experiments.exp7_memory_bottleneck
```

포트 수를 훑으며 **병목이 메모리에서 연산으로 넘어가는 지점**을 찾습니다.
훑을 범위는 `config/sweeps.yaml`의 `exp7`에서 바꿉니다.

---

## 7. 결과 읽기

### CSV 열기

```bash
PYTHONUTF8=1 python -c "
import csv, io
r = list(csv.DictReader(io.open('outputs/raw/exp2_margin_sweep.csv', encoding='utf-8')))
print(len(r), '행')
print(list(r[0].keys()))
for x in r[:5]: print({k: x[k] for k in ('margin','read_saving_bram','top16_retention')})"
```

엑셀로 열어도 됩니다. **UTF-8이라 "데이터 → 텍스트 나누기"로 인코딩을 UTF-8로 지정**해야 한글이 안 깨집니다.

### 꼭 봐야 하는 열

| 열 | 뜻 | 주의 |
|---|---|---|
| `read_saving_bram` | **실현되는** 읽기 절감 | `read_saving_ideal`을 쓰면 안 됩니다 |
| `top16_retention` | 참 top-16 보존율 | 1.0이면 무손실 |
| `total_cycles` | **연산** 사이클만 | 이것만 보면 항상 진다 |
| `total_cycles_with_memory` | 메모리 포함 | **이게 진짜 시간** |
| `mean_survivor_frac` | 평균 생존 비율 | 낮을수록 좋다 |
| `decision_latency_mean` | 평균 판정 지연(평면) | `config`에 있습니다 |

> **`read_saving_ideal`과 `read_saving_bram`이 벌어지는 게 정상입니다.**
> 앞은 이론값, 뒤는 BRAM 워드 단위로 실제 읽어야 하는 양입니다.
> 논문에는 **뒤쪽만** 씁니다.

---

## 8. 자주 나는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| 한글이 `����`로 | cp949 | `PYTHONUTF8=1` |
| `출처: synthetic` | 캡처 파일명 불일치 | §4-2 |
| crosscheck 불일치 | mock이 옛 설정 | `--generate-mock` 후 재실행 |
| `score_idx N이 9b를 넘는다` | `N_MAX` 올리고 `idx_bits` 안 올림 | `hardware.yaml`의 `idx_bits` |
| 모델 다운로드 시도 | 오프라인 미지정 | `HF_HUB_OFFLINE=1` |
| `ConfigDefaultWarning` | yaml 절 이름 오타 | 경고에 유사 키를 알려 줍니다 |
| 스윕 범위가 이상함 | **yaml 리스트를 여러 줄로 썼다** | 아래 |
| exp4가 안 끝남 | T=2048에서 300조합 | `sweeps.yaml`에서 축을 줄이거나 `--quick` |
| 메모리 부족 | 부분내적 텐서 | `seq_len`을 낮추거나 `chunk` 사용 |
| **로딩 중 Segmentation fault** | **RAM 부족.** Llama 1B fp32는 약 5 GB | 아래 |

### yaml 리스트는 반드시 한 줄로

PyYAML이 없을 때 쓰는 **자체 파서가 여러 줄 리스트를 못 읽고 뒷줄을 조용히 버립니다.**

```yaml
# ✗ 이렇게 쓰면 PyYAML 유무에 따라 스윕 범위가 달라진다
margin: [0.0, 0.2, 0.4, 0.5, 0.6,
         0.7, 0.8, 0.9, 1.0]

# ✓ 한 줄로
margin: [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

실제로 겪었습니다 — `exp2`의 margin이 PyYAML로는 17점, 자체 파서로는 11점이 됐습니다.
`tests/run_tests.py`가 잡아 주지만, 애초에 한 줄로 쓰는 게 낫습니다.

```bash
# 두 파서가 같은 결과를 내는지 직접 확인
PYTHONUTF8=1 python -c "
import io, yaml
from src.config import _mini_yaml_load
t = io.open('config/sweeps.yaml', encoding='utf-8').read()
print('일치:', yaml.safe_load(t) == _mini_yaml_load(t))"
```

### 로딩 중 Segmentation fault

가장 자주 겪은 문제입니다. **모델 로딩이 70% 근처에서 죽고 exit code 139**가 나옵니다.

```bash
# 남은 메모리 확인
PYTHONUTF8=1 python -c "
import ctypes
class MS(ctypes.Structure):
    _fields_=[('l',ctypes.c_ulong),('mL',ctypes.c_ulong),('tp',ctypes.c_ulonglong),
              ('ap',ctypes.c_ulonglong),('tpf',ctypes.c_ulonglong),('apf',ctypes.c_ulonglong),
              ('tv',ctypes.c_ulonglong),('av',ctypes.c_ulonglong),('ae',ctypes.c_ulonglong)]
s=MS(); s.l=ctypes.sizeof(MS)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
print(f'전체 {s.tp/1e9:.1f}GB  사용가능 {s.ap/1e9:.1f}GB')"
```

**사용 가능이 6 GB 미만이면 fp32 로딩이 실패합니다.**

| 해결 | 방법 |
|---|---|
| 동시 실행 금지 | **모델을 쓰는 실험을 두 개 동시에 돌리지 마세요.** 세그폴트의 가장 흔한 원인입니다 |
| 브라우저·IDE 닫기 | 6 GB는 금방 채워집니다 |
| `exp9`는 bf16 | 이미 `bfloat16 + low_cpu_mem_usage`로 로드합니다 (약 2.5 GB) |
| `exp8`·`model_hooks`는 fp32 | perplexity는 정밀도가 필요합니다. 메모리를 비우고 돌리세요 |

> `exp9`가 bf16을 쓰는 게 괜찮은 이유 — 뽑은 q/k는 **곧바로 INT8로 양자화**됩니다.
> bf16 정밀도면 충분합니다. `exp8`은 perplexity 자체를 재므로 fp32를 씁니다.

---

## 9. 논문 수치를 뽑는 순서

```bash
# 1. 설정 확정
vim config/model.yaml config/quant.yaml config/hardware.yaml

# 2. 깨진 데 없는지
PYTHONUTF8=1 python tests/run_tests.py

# 3. 실제 텐서 캡처 (설정의 seq_len 과 같은 값으로)
PYTHONUTF8=1 HF_HUB_OFFLINE=1 python -m src.model_hooks \
    --seq-len 2048 --layer 8 --head 0 --rope --text-file cache/wikitext2_test.txt

# 4. 실제 텐서가 잡히는지 확인
PYTHONUTF8=1 python -c "
from src.config import load_config; from src.dataset import snapshot_from_config
print(snapshot_from_config(load_config(), seq_len=2048).source)"

# 5. mock 재생성 + crosscheck
PYTHONUTF8=1 python run_paper_experiments.py --generate-mock
PYTHONUTF8=1 python run_paper_experiments.py --crosscheck

# 6. 전체 실행
PYTHONUTF8=1 python run_paper_experiments.py --no-figures

# 7. 무거운 둘은 따로. ★ 반드시 하나씩 ★ 동시에 돌리면 세그폴트가 난다
PYTHONUTF8=1 HF_HUB_OFFLINE=1 python -m experiments.exp8_real_perplexity --tokens 2048
PYTHONUTF8=1 HF_HUB_OFFLINE=1 python -m experiments.exp9_margin_coverage --heads 8 --layers 4
```

> ### 논문에 쓸 때 지킬 것
>
> - **`source: estimate`인 값은 그대로 쓰지 마세요.** Vivado 실측으로 교체할 자리입니다.
>   `run_paper_experiments.py`가 실행할 때마다 몇 개인지 경고합니다(현재 8개).
> - **`read_saving_bram`을 쓰세요.** `ideal`은 실현되지 않는 값입니다.
> - **`total_cycles_with_memory`를 쓰세요.** `total_cycles`는 연산축만입니다.
> - **`--quick` 결과를 쓰지 마세요.** 짧은 문맥은 판정 지연 때문에 절감이 반토막입니다.
> - **perplexity는 exp8 실측을 쓰세요.** 보간값(+5.8%)은 우리에게 불리한 추정이었습니다.

---

## 10. 팀 1에게 넘길 때

```bash
git apply team2_code/onto_team1.patch
PYTHONUTF8=1 python tests/run_tests.py        # 424 통과
```

`team2_code/README.md`에 무엇을 가져왔고 무엇을 바꿨는지 정리돼 있습니다.
