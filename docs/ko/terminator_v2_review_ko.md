# `terminator_v2` 검토 결과 — 채택했습니다

**2026-08-29 · 팀 2 (정원탁 · 채지훈)**

먼저 결론부터 — **거의 그대로 채택했습니다.** 실제로 고친 건 기본값 5줄입니다.

```
   우리 원본     155줄
   받은 v2       441줄
   채택본        467줄       v2 대비  +36 / -10 줄  (그중 코드는 10줄, 나머지는 주석)
```

`src/terminator.py`를 v2로 교체하고, `tests/test_terminator.py`를 22개 → **30개**로 늘렸습니다.
전체 테스트가 **416개 → 424개** 통과합니다.

---

## 1. 어떻게 검증했나

동작이 바뀌면 무손실 보장이 깨지므로, **읽어서 판단하지 않고 수치로** 확인했습니다.

### 1-1. 등가성 — 5,076건

옛 구현과 v2에 **같은 입력을 넣고 출력을 전부 비교**했습니다.

| 무엇 | 조합 | 결과 |
|---|---|---|
| 무작위 | 5,000회 — θ정책 4종 × margin 4모드 × 지연 0~8 × `top_k` 7종 × 종단 on/off | **불일치 0** |
| 경계 | 64조합 — `T=1` / `T<k` / `T=k` / `q` 전부 0·양수·음수 | **불일치 0** |
| 극단 | 12조합 — `P` 전부 0 / 최대 / 최소 / 부호 교차 | **불일치 0** |

비교한 필드: `s_int` · `alive` · `term_plane` · `read_live` · `live_count` · `theta_trace` · `theta_final`
**전부 비트 단위로 일치합니다.**

### 1-2. 기존 테스트

```
   test_terminator     22개  통과
   test_designs        12개  통과      (통합 경로)
   test_decode_loop    18개  통과
   전체               416개  통과 / 0 실패
   crosscheck          12/12
```

> 중간에 실패 1건이 났는데 **v2 결함이 아니었습니다.** 제가 비교용으로 `src/`에
> 파일 두 개를 남겨 둬서 팀 1의 의존성 테스트가 그걸 센 것입니다. 정리 후 통과했습니다.

### 1-3. 새 API가 실제로 맞는지

`TerminationController`는 새 경로이므로 따로 확인했습니다.

| 검사 | 결과 |
|---|---|
| 스트리밍(`process_plane` 반복) == 배치(`run_step`) · 300회 | 불일치 0 |
| `lower <= upper` 항상 | OK |
| 구간이 평면마다 단조 수축 | OK |
| 마지막 평면에서 `lower == upper` (점으로 수렴) | OK |
| `killed ⊆ live_before` | OK |
| `live_after == live_before & ~killed` | OK |
| `theta` 단조 증가 | OK |
| 죽은 토큰의 `upper < theta + margin` | OK |
| `run_step_from_frontend`가 `Q+/Q-`를 맞게 계산 · 200회 | 불일치 0 |
| 입력 검증 7종이 실제로 막는가 | 전부 OK |

### 1-4. 돌연변이 시험

코드를 일부러 망가뜨리고 테스트가 잡는지 셌습니다. **통과만으로는 근거가 안 되기 때문입니다.**

```
   돌연변이                    v2      우리원본
   M2 l_offset 부호 뒤집기     O        O
   M6 판정 지연 무시           O        O
   M7 keep-top-k 가드 제거     O        O
   M8 upper < theta -> <=     O        O
   M1 r(m) 계수 +1            X        X      <- 두 버전 공통 구멍
   frozen 값 파괴              X        X      <- 두 버전 공통 구멍
```

**검출력이 완전히 같습니다.** 뒤의 두 구멍은 v2 문제가 아니라 **원래 있던 것**입니다(§4).

### 1-5. 성능

`run_decode`(T=512, 480스텝)를 **번갈아 9회** 재고 중앙값을 봤습니다.

```
   우리 원본   0.735s
   v2         0.862s        1.17배
```

평면마다 `PlaneDecision`을 만들며 배열을 6개 복사하는 비용입니다.
**받아들일 만합니다** — 얻는 것이 더 큽니다.

> 처음에 단발로 재서 **4.33배**가 나왔는데 잡음이었습니다. 그 숫자로 판단할 뻔했습니다.

---

## 2. 무엇을 가져왔나 — 세 가지가 값어치 있었습니다

### ★ (1) `TerminationController` — 평면별 관측

이게 v2의 핵심입니다. 우리 `run_step`은 8평면을 통째로 돌고 **결과만** 줍니다.

```python
ctrl = TerminationController(bounds, policy, n_tokens, decision_latency=1)
while not ctrl.done:
    dec = ctrl.process_plane(P[ctrl.next_plane])   # PlaneDecision
result = ctrl.finish()
```

`PlaneDecision`이 평면마다 **`read_mask` · `live_before` · `live_after` · `killed` · `lower` · `upper` · `theta` · `margin_abs`** 를 전부 내줍니다.

**왜 필요한가** — `ARCHITECTURE.md` §10 검증 계획이 "골든 벡터"를 요구합니다.
RTL 파형과 대조하려면 **평면마다** 중간값을 봐야 하는데, 기존 구현으로는 못 봅니다.
이 인터페이스가 그걸 바로 가능하게 합니다.

### ★ (2) `run_step_from_frontend` — 팀1 → 팀2 인계를 한 함수로

```python
run_step_from_frontend(q_stored, partials, ...)
#   내부에서 step_bounds(q_stored) 로 Q+/Q- 를 스텝 시작 시 딱 한 번 계산
```

기존에는 호출자가 `StepBounds`를 직접 만들어야 해서 **팀 1이 우리 내부를 알아야** 했습니다.
이 함수가 경계를 코드로 명시합니다. `팀분업_인계.md`의 "인계 2"가 바로 이 지점입니다.

### ★ (3) 평면수 불일치 차단 — 실제 결함을 하나 막았습니다

이게 가장 값어치 있었습니다.

```
   partials 4평면  +  bounds n_planes=8

   우리 원본   예외 없음  ← ★ 조용히 틀린 답 (생존 10개를 그냥 반환)
   v2         ValueError: plane-count mismatch ... partials=4, bounds=8
```

반대 방향(8평면 + `n_planes=4`)은 우리 것도 `IndexError`로 죽지만,
**평면이 적을 때는 그냥 통과**합니다. `l_offset(m)`이 다른 가중치를 써서 상한식이
성립하지 않는데도 답이 나옵니다.

팀 1이 `n_planes`를 다르게 준 채 넘기면 못 잡는 상황이었습니다.

그 외 `n_tokens < 0`, `decision_latency < 0`, 평면 shape 불일치, `finish()` 조기 호출,
9번째 평면 호출도 전부 막아 주는 것 확인했습니다.

---

## 3. 하나 고쳤습니다 — `run_step_from_frontend`의 기본값

**이것만 손댔습니다.**

```python
# v2 원본
top_k: int = 8,
theta_policy: str = "every_plane",
once_at_m: int = 3,
margin: float = 0.0,
margin_mode: str = "relative_gap",       # ← 여기

# 채택본
top_k: int | None = None,                # None 이면 ThetaPolicy() 기본값을 따른다
...
d = ThetaPolicy()
policy = ThetaPolicy(
    name=d.name if theta_policy is None else theta_policy,
    top_k=d.top_k if top_k is None else int(top_k),
    ...
)
```

### 왜 고쳤나

셋 다 **우리가 이미 고친 값의 옛 버전**입니다.

| 항목 | v2 기본값 | 확정값 | 어디서 정했나 |
|---|---|---|---|
| `margin_mode` | `relative_gap` | **`relative_width`** | `ARCHITECTURE.md`, `ThetaPolicy` 기본값까지 변경 |
| `top_k` | 8 | **16** | `exp8` 실측 perplexity |
| `once_at_m` | 3 | **4** (M0) | 생존 곡선상 평면 4부터 종단 시작 |

특히 `relative_gap`이 문제였습니다. 저희가 `relative_width`로 확정하면서 `ThetaPolicy`
기본값까지 바꿨는데, **이 편의 함수가 그걸 우회해서 되돌리고 있었습니다.**

`run_step_from_frontend(q, p)`를 기본값으로 부르면 확정 설정이 아닌 옛 설정으로 도는 셈입니다.

> 같은 종류의 사고를 전에 한 번 겪었습니다. `generate_mock`이 `word_tokens: 32`를 키에
> 하드코딩해서, config를 1로 바꿔도 mock만 옛 설정으로 생성됐습니다.
> 게다가 `predict_for`가 같은 키를 써서 **crosscheck가 통과해 버려** 불일치가 안 드러났습니다.
>
> **편의 함수가 자체 기본값을 들고 있으면 설정이 조용히 무시됩니다.**

`margin_mode`를 되돌리는 돌연변이를 넣어 테스트가 잡는지도 확인했습니다 — 잡습니다.

---

## 4. 두 버전 공통 구멍 두 개 — 이건 원래 있던 것입니다

v2 검증 중에 **우리 원본에도 있던** 테스트 구멍이 드러났습니다. 함께 메웠습니다.

### `frozen` — 종단된 토큰의 확정 점수

```
   frozen[killed] = 0 으로 바꿔도  →  416개 테스트 중 0개 실패
```

`frozen`은 **종단 시점의 부분 점수 `S_m`** 입니다. 확인해 보니

- `output_stage`는 생존만 내보내므로 안 샙니다
- 그런데 **`designs.DesignResult.scores_raw`로 그대로 노출됩니다**
- 기존 테스트는 전부 **생존 토큰만** 보거나 두 실행을 비교합니다

**종단 토큰의 값이 무엇이어야 하는지 못 박은 테스트가 없었습니다.**

테스트 2개를 추가했습니다.

```python
test_terminated_token_keeps_its_partial_score_at_termination
    종단 토큰의 s_int == cumulative_accumulate(p)[term_plane]  (전 토큰 확인)
    생존 토큰의 s_int == 최종 누산값

test_terminated_score_is_neither_zero_nor_final
    0 으로 두는 것과 최종값으로 두는 것을 양쪽 다 막는다
```

### `M1` — `r(m)`에 +1

`test_terminator`만으로는 안 잡히지만 `bounds`·`quantize`·`threshold` 쪽에서 잡힙니다.
**단일 파일 검출력의 문제이지 전체 구멍은 아닙니다.**

---

## 5. 추가한 테스트 8개

```
   test_streaming_equals_batch                       스트리밍 == 배치 (40회 무작위)
   test_plane_decision_reports_what_actually_...     PlaneDecision 정합성 6가지
   test_controller_rejects_misuse                    오용 5종 차단
   test_frontend_computes_bounds_once_and_...        Q+/Q- 계산 (30회)
   test_frontend_defaults_follow_theta_policy_...    ★ 기본값 우회 방지
   test_plane_count_mismatch_is_rejected             ★ 조용히 틀린 답 차단
   test_terminated_token_keeps_its_partial_score...  ★ frozen = 종단 시점 S_m
   test_terminated_score_is_neither_zero_nor_final   ★ 0도 최종값도 아니다
```

### 시험 하나는 처음에 무력했습니다

`test_frontend_defaults_...`를 `margin=0`으로 썼더니 **`relative_gap`과 `relative_width`가
같은 답**을 내서 아무것도 못 막았습니다(margin이 0이면 두 모드 다 0을 냅니다).

두 모드가 실제로 갈리는 조건(`seed=0, T=50, k=8, margin=0.5`)을 찾아 넣고,
**갈림 자체를 단언**해서 조건이 무뎌지면 걸리게 했습니다.

```python
assert not np.array_equal(auto.alive, wrong.alive), (
    "relative_gap 과 relative_width 가 같은 답을 내면 이 시험이 무의미하다"
)
```

---

## 6. 최종 상태

```
   테스트        424 통과 / 0 실패      (416 -> 424)
   crosscheck    12/12
   수치 회귀      없음
```

적용 전후 수치가 정확히 같습니다.

```
     설계  margin      절감       보존     생존
    exact    0.00   24.61%   1.00000   11.3%     <- 적용 전과 동일
   approx    0.40   32.64%   1.00000   10.4%
   approx    0.70   44.98%   1.00000    9.9%
```

돌연변이 검출은 **6개 → 10개**로 늘었습니다.
남은 하나(`frontend top_k` 되돌림)는 `ThetaPolicy().top_k`가 이미 8이라 **의미 없는 돌연변이**입니다.

---

## 7. 정리

**좋았던 것**

- **기존 `run_step` API를 그대로 유지**한 게 결정적이었습니다. `designs.py`·`decode_loop.py`·
  experiments를 하나도 안 고치고 교체됐습니다.
- 평면수 불일치 검증이 **실제 결함을 막았습니다.** 우리 것은 조용히 틀린 답을 냈습니다.
- `PlaneDecision`이 내주는 값들이 전부 정합했습니다(6가지 성질 확인). RTL 대조에 바로 씁니다.

**앞으로 참고하실 것**

- **편의 함수에 기본값을 두면 설정이 우회됩니다.** 확정값이 여러 곳에 흩어지면 언젠가 갈라집니다.
  `ThetaPolicy` 같은 단일 출처를 두고 거기서 가져오는 편이 안전합니다.
- 성능은 1.17배 느려집니다. `PlaneDecision`이 필요 없는 경로(스윕 등)에서는 안 만들도록
  옵션을 두는 것도 방법입니다 — 지금은 그대로 두었습니다.

---

*채택본은 `공유_2026-08-29/코드/src/terminator.py`, 테스트는 같은 폴더 `tests/test_terminator.py`입니다.*
*확정 파라미터와 그 근거는 `문서/ARCHITECTURE.md`, 두 팀 경계는 `문서/팀분업_인계.md`를 참고해 주세요.*
