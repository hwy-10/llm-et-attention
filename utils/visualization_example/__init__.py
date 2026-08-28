"""행렬 곱 시각화 데모 — 계산은 파이썬, 그리기는 브라우저.

교보재다. 코어 시뮬레이터는 이 패키지를 임포트하지 않는다.

  python -m utils.visualization_example

A 의 i 행과 B 의 j 열을 고르면 C[i][j] 가 만들어지는 과정을 항 단위로
펼쳐 보여 준다. 어텐션의 ``q · K^T`` 도 결국 같은 모양이라, 팀에 새로
합류한 사람에게 "우리가 가속하는 그 한 줄"을 설명할 때 쓰려고 만들었다.

구성
----
``matmul.py``   계산 전부. 곱셈·항별 분해·numpy 교차 검증.
``depgraph.py`` 담당 범위의 코드 상관관계 — 저장소를 ast 로 파싱한다.
``server.py``   표준 라이브러리 http.server 기반 웹 서버 + JSON API.
``static/``     화면. 산술은 한 줄도 하지 않고 서버가 준 값을 그리기만 한다.
"""

from . import depgraph, matmul, server

__all__ = ["depgraph", "matmul", "server"]
