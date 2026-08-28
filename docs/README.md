# docs/ — 이 프로젝트의 문서 전부

**루트에는 [README.md](../README.md) 하나만 둔다.** 재현 방법과 실험 목록은 거기 있고,
**배경·설계·구조·선행조사는 전부 여기**다.

## 목차

| 문서 | 답하는 질문 | 언제 |
|---|---|---|
| **[background/](#background--llm-어텐션-기초)** | 이걸 이해하려면 **무엇을 미리 알아야 하는가** | 배경이 없다면 가장 먼저 |
| [architecture.md](architecture.md) | 우리가 **무엇을 만드는가** | 설계를 이해할 때 |
| [structure.md](structure.md) | 코드가 **어디에 있는가** | 코드를 만지기 직전 |
| [related_work.md](related_work.md) | 남들은 **이미 무엇을 했는가** | 왜 이걸 하는지 · 발표 준비 |

읽는 순서는 표의 위에서 아래다. 배경 없이 `architecture.md` 부터 열면
§2 양자화 규약에서 막힌다.

---

## 영역 폴더

### `background/` — LLM 어텐션 기초

하드웨어를 하는 사람이 **"왜 하필 Q·K^T 인가"** 에 답할 수 있게 하는 것이 목표다.
학습(training)이나 모델 구조 전반은 다루지 않는다.

| 문서 | 무엇을 | 언제 읽나 |
|---|---|---|
| [transformer.md](background/transformer.md) | Q/K/V, 어텐션 4단계, GQA, **Prefill vs Decode**, KV 캐시 | **가장 먼저.** 배경이 없다면 필수 |
| [attention_walkthrough.md](background/attention_walkthrough.md) | 4토큰 수치 예제 — 점수 계산부터 **비트평면 종단까지** 손으로 | transformer.md 다음. 설계가 손에 안 잡힐 때 |
| [llama_3_2_1b.md](background/llama_3_2_1b.md) | 대상 모델 스펙·성능, **KV 캐시 크기 계산** | 단일 헤드로 범위를 좁힌 이유가 궁금할 때 |

**추천 경로**

```
transformer.md §5 (Prefill vs Decode)        ← 프로젝트 전체의 전제
        ↓
attention_walkthrough.md §3~4                 ← 우리가 계산하는 그 한 줄
        ↓
architecture.md §1~3                          ← 설계
```

**어디부터 읽어도 되지만, `transformer.md §5` 만은 건너뛰지 말 것.**
"어텐션은 메모리 중심 가속기에 부적합하다"는 흔한 서술이 **Prefill 이야기**라는 것을
모르면, 이 프로젝트가 왜 성립하는지 자체가 안 보인다.

**활용 팁**

* **팀에 새로 합류한 사람** — `transformer.md` → `attention_walkthrough.md` 순으로 하루면 된다.
  NLP 배경이 아예 없어도 따라올 수 있게 썼다.
* **손으로 확인하고 싶다면** — `attention_walkthrough.md §5` 의 코드 10줄을 돌려 보면 된다.
  그 안의 `assert` 가 [tests/test_bounds.py](../tests/test_bounds.py) 가 강제하는 불변식과 같다.
* **발표를 준비한다면** — 세 문서의 ★ 표시 절이 반론 방어점이다.
  Prefill/Decode 구분, KV 캐시 크기, 상한식이 서는 자리.
* **RTL 을 짠다면** — `llama_3_2_1b.md §4` 의 KV 캐시 산술만 봐도 된다.
  나머지는 [config/hardware.yaml](../config/hardware.yaml) 과 [rtl_data/schema.md](../rtl_data/schema.md) 쪽이다.

---

## 단일 문서

폴더를 만들 만큼 갈래가 없는 것은 `docs/` 바로 밑에 파일로 둔다.

| 문서 | 무엇을 | 어디부터 보면 되나 |
|---|---|---|
| [architecture.md](architecture.md) | 조감도 해설 — 대상 · 양자화 규약 · 제안 설계 · 검증 · 결과 | **§2 양자화 규약.** 상한식이 서는 자리라 나머지 전부의 전제다 |
| [structure.md](structure.md) | 파일이 왜 이렇게 배치되어 있는가 + 저장소 트리 | §0 설계 원칙 네 가지. 코드를 만지기 직전에 |
| [related_work.md](related_work.md) | 같은 기법을 먼저 한 논문 조사 (2026-08) | §0. **PADE 가 이미 존재한다**는 사실부터 본다 |

셋 다 그림은 `slides/` 에 있고 문서에서 상대경로로 부른다
(`architecture.md` → `../slides/architecture/`).

---

## 문서를 추가할 때

**먼저 폴더인지 파일인지 정한다.**

| 갈래가 | 어디에 | 지금 예 |
|---|---|---|
| 여러 문서로 나뉜다 | `docs/<영역>/` 폴더 | `background/` — 3개 |
| 하나로 끝난다 | `docs/<이름>.md` 파일 | `architecture.md` · `structure.md` · `related_work.md` |

애매하면 **파일로 두고 나중에 옮긴다.** 저 셋도 원래 루트에 있다가 여기로 내려온 것이다.
RTL·합성 단계로 넘어가면 Verilog·Vivado 입문이나 FPGA 자원 읽는 법 같은 것이 붙을 텐데,
성격이 다르니 `background/` 에 섞지 말고 새 폴더로 두는 편이 낫다.

**규칙 네 가지.**

1. **파일 이름은 소문자.** `docs/` 안은 전부 소문자다. 대문자는 루트의
   `README.md`·`LICENSE` 관례이지 여기 관례가 아니다.
2. **그림은 여기 두지 않는다.** `slides/` 에 두고 부른다 — 영역 폴더는
   `slides/docs_<영역>/`, 단일 문서는 `slides/<문서이름>/`
   (`background/` → `slides/docs_background/`, `architecture.md` → `slides/architecture/`).
   규약은 [slides/README.md](../slides/README.md).
3. **위 "목차" 표에 한 줄 넣는다.** 영역 폴더면 절도 하나 추가한다 —
   문서 표 + 추천 경로 + 활용 팁. 폴더 안 문서가 5개를 넘어가면 그때 그 폴더에
   자체 README 를 두고 여기서는 링크만 남긴다.
4. **링크는 그 문서 파일 위치 기준 상대경로.** 저장소 최상위 기준이 아니다.
   `docs/` 밑이면 `../src/…`, `docs/background/` 밑이면 `../../src/…` 가 된다.

[structure.md](structure.md) 의 트리에도 반영할 것.
