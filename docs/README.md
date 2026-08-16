# docs/ — 배경지식

**이 프로젝트를 이해하는 데 필요한 선행 지식**을 모아 둔 곳이다.
설계 자체가 아니라 **설계를 읽기 위한 준비물**이다.

세 곳이 헷갈리기 쉬우니 먼저 구분한다.

| | 답하는 질문 |
|---|---|
| **docs/** (여기) | 이걸 이해하려면 **무엇을 미리 알아야 하는가** |
| [architecture.md](../architecture.md) | 우리가 **무엇을 만드는가** |
| [STRUCTURE.md](../STRUCTURE.md) | 코드가 **어디에 있는가** |

읽는 순서는 저 표의 위에서 아래다. 배경 없이 architecture.md 부터 열면
§2 양자화 규약에서 막힌다.

---

## 영역 목록

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

## 앞으로 영역이 늘어나면

지금은 `background/` 하나지만 RTL·합성 단계로 넘어가면 늘어날 수 있다.
예를 들어 Verilog·Vivado 입문이나 FPGA 자원 읽는 법 같은 것은 성격이 달라
`background/` 에 섞지 말고 별도 폴더로 두는 편이 낫다.

**규칙 세 가지만 지키면 된다.**

1. **한 영역 = 한 폴더.** `docs/<영역>/`
2. **그림은 여기 두지 않는다.** `slides/docs_<영역>/` 에 두고 문서에서 상대경로로 참조한다
   (예: `background/` → `slides/docs_background/`). 규약은 [slides/README.md](../slides/README.md).
3. **이 파일의 "영역 목록"에 한 절을 추가한다.** 문서 표 + 추천 경로 + 활용 팁.
   폴더 안 문서가 5개를 넘어가면 그때 그 폴더에 자체 README 를 두고 여기서는 링크만 남긴다.

[STRUCTURE.md](../STRUCTURE.md) 의 트리에도 반영할 것.
