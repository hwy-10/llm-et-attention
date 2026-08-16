# rtl_data/ — RTL 팀 ↔ 소프트웨어 팀 데이터 계약

이 디렉토리는 **RTL/FPGA 쪽에서 소프트웨어 쪽으로 넘어오는 유일한 경로**다.
여기 파일 형식만 지키면 두 팀이 서로를 기다리지 않고 동시에 진행할 수 있다.

RTL 이 아직 없어도 `mock/` 의 더미 데이터로 파이프라인 전체가 돌아간다.
실측이 들어오면 이 디렉토리 **최상위**에 같은 이름으로 놓기만 하면 된다
(최상위 파일이 있으면 mock 대신 그쪽을 읽는다).

`mock/rtl_simulation_cycles.csv` 는 소프트웨어 모델에서 생성한 것이라
`--crosscheck` 가 항상 통과한다. 배관이 살아 있는지 확인하는 용도다.
설정을 바꾼 뒤에는 `python run_paper_experiments.py --generate-mock` 으로 다시 만든다.

---

## 1. `rtl_simulation_cycles.csv` — 시뮬레이션 실측

테스트벤치가 조건별로 한 줄씩 뱉는다. **필수 열이 하나라도 없으면 파서가 예외를 던진다**
(조용히 잘못된 값이 논문에 들어가는 것을 막기 위해서다).

| 열 | 타입 | 의미 |
|---|---|---|
| `design` | str | `baseline` \| `seq` \| `exact` \| `approx` — src/designs.py 와 같은 이름 |
| `seq_len` | int | 문맥 길이 (활성 토큰 수의 최종값) |
| `top_k` | int | 상위 k |
| `margin` | float | θ 여유값. 정확 모드는 0.0 |
| `cycles` | int | 디코드 루프 전체의 총 사이클 |
| `bram_reads` | int | **BRAM 워드 읽기 횟수** (토큰 수가 아니라 워드 수) |

선택 열 (있으면 대조에 함께 쓰인다):

| 열 | 의미 |
|---|---|
| `mean_term_plane` | 평균 종단 평면 |
| `n_steps` | 디코드 스텝 수 |
| `stall_cycles` | 파이프라인 빈 구간 사이클 |
| `word_tokens` | 한 BRAM 워드에 담긴 토큰 수 |

> ⚠ `bram_reads` 를 "읽은 토큰 수"로 세면 안 된다. 소프트웨어 모델은
> **워드 단위**로 회계하므로, 단위가 다르면 대조가 통째로 어긋난다.
> src/memory.py 상단 주석 참조.

`utils/crosscheck.py` 가 `(design, seq_len, top_k, margin)` 을 키로 예측과 대조하고,
오차가 허용치(기본 5%)를 넘으면 원인 후보를 함께 출력한다.

---

## 2. 합성 보고서

파일명 규약: `<design>_synth.rpt`, `<design>_timing.rpt`, `<design>_power.rpt`

`<design>` 은 `config/hardware.yaml` 의 `resources` 키와 같아야 한다:
`baseline`, `seq_no_et`, `exact_et`, `approx_et`

`utils/hw_parser.py` 가 다음을 뽑는다.

| 보고서 | 추출 항목 |
|---|---|
| `report_utilization` | Slice LUTs, Slice Registers, DSPs, Block RAM Tile |
| `report_timing_summary` | WNS(ns), 클럭 주기 → **Fmax = 1/(T − WNS)** |
| `report_power` | Dynamic (W), Device Static (W) |

### ★ 권장: JSON 요약을 직접 뱉을 것 ★

Vivado 보고서 포맷은 버전마다 달라 정규식 파서가 잘 깨진다.
Tcl 에서 아래 형식의 `<design>.json` 을 직접 쓰면 훨씬 안전하다.
`hw_parser.load_report()` 가 `.json` 을 우선 인식한다.

```json
{
  "design": "exact_et",
  "lut": 15600,
  "ff": 8900,
  "dsp": 0,
  "bram36": 16,
  "clock_period_ns": 6.0,
  "wns_ns": 0.34,
  "fmax_mhz": 176.7,
  "dynamic_power_mw": 168.0,
  "static_power_mw": 105.0,
  "vivado_version": "2023.2",
  "part": "xc7z020clg400-1"
}
```

Tcl 예시:

```tcl
set fp [open "rtl_data/${design}.json" w]
puts $fp "{"
puts $fp "  \"design\": \"$design\","
puts $fp "  \"lut\": [get_property SLICE_LUTS [get_cells -hier]],"
...
close $fp
```

---

## 3. 값이 들어오면 할 일

1. 실측 파일을 `rtl_data/` 최상위에 놓는다.
2. `config/hardware.yaml` 의 `resources.*` 를 실측값으로 갱신하고
   `source` 를 `estimate` → `vivado_synth` 로 바꾼다.
3. `python run_paper_experiments.py --crosscheck` 로 예측과 대조한다.
4. 불일치가 나오면 **소프트웨어 모델이 틀린 것**이다.
   `decision_latency_planes`, `word_tokens`, `lanes`, `compaction_cost_cycles`
   를 실제 RTL 파라미터에 맞춘다.
5. `cfg.provenance_warnings()` 가 빈 목록이 되면 논문에 쓸 준비가 된 것이다.
