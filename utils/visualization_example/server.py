"""행렬 곱 시각화 데모의 웹 서버.

표준 라이브러리만 쓴다 — Flask·FastAPI 를 넣지 않은 것은 취향이 아니라
저장소 규약이다. ``requirements.txt`` 는 numpy 하나만 필수로 두고 나머지는
"없어도 코어가 돌아가는" 선택 의존성으로 관리한다. 교보재 하나 때문에
웹 프레임워크를 필수 목록에 올릴 이유가 없다.

역할 분담
--------
* **파이썬** — 곱셈·누산·검증 (:mod:`utils.visualization_example.matmul`) 과
  스케줄 정책 (:mod:`src.schedule` 을 그대로 호출한다)
* **브라우저** — 그리기와 클릭 처리. 산술은 한 줄도 하지 않는다.

페이지는 둘이다::

    /            행렬 곱 — A 의 행 x B 의 열 = C 의 칸
    /schedule    조기 종단 스케줄 — 사이클 감소 vs BRAM 워드 감소
    /schedule_py src/schedule.py 함수 해부 — 입력 → 핵심 동작 → 출력
    /glossary    용어 사전 — 같은 격자 위에 겹쳐 본다
    /depgraph    SW 검증 담당 범위의 코드 상관관계 (import 그래프)

브라우저가 A·B 를 들고 있다가 매번 통째로 보내고, 서버는 상태를 남기지
않는다. 새로고침하면 값이 새로 뽑히는 대신, 여러 명이 동시에 열어도
서로 간섭하지 않는다.

실행
----
::

    python -m utils.visualization_example            # 브라우저까지 열림
    python -m utils.visualization_example --port 8123 --no-browser
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import anatomy, depgraph, glossary, matmul, schedule_demo

__all__ = ["build_state", "make_handler", "main", "serve"]

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY = 256 * 1024  # 12×12 행렬 두 개면 수 KB 다. 넉넉하되 무제한은 아니게.

DEFAULT_DIMS = (5, 4, 3)  # (M, N, P) — README 와 architecture 예시에 맞춘다


# ---------------------------------------------------------------------------
# 응답 조립
# ---------------------------------------------------------------------------

def build_state(A: Any, B: Any, i: Any = None, j: Any = None) -> dict[str, Any]:
    """화면 한 장을 그리는 데 필요한 값을 전부 계산해 돌려준다.

    선택된 칸이 없으면(``i`` 또는 ``j`` 가 None) ``cell`` 은 None 이다.
    """
    a, b = matmul.validate_pair(A, B)
    c = matmul.multiply(a, b)

    cell = None
    if i is not None and j is not None:
        cell = matmul.breakdown(a, b, int(i), int(j))

    return {
        "A": a,
        "B": b,
        "C": c,
        "dims": {"m": len(a), "n": len(b), "p": len(b[0])},
        "cell": cell,
        "backend": matmul.backend_name(),
    }


def _random_state(m: int, n: int, p: int, seed: int | None = None) -> dict[str, Any]:
    a = matmul.random_matrix(m, n, seed=seed)
    b = matmul.random_matrix(n, p, seed=None if seed is None else seed + 1)
    return build_state(a, b)


# ---------------------------------------------------------------------------
# 핸들러
# ---------------------------------------------------------------------------

def make_handler(quiet: bool = False) -> type[BaseHTTPRequestHandler]:
    """요청 핸들러 클래스를 만든다 (``quiet`` 면 접속 로그를 끈다)."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "MatmulViz/1.0"
        protocol_version = "HTTP/1.1"

        # -- 유틸 ----------------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def _fail(self, code: int, message: str) -> None:
            self._json(code, {"error": message})

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            if length > MAX_BODY:
                raise ValueError(f"request body too large ({length} bytes)")
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        # -- 라우팅 --------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 - stdlib 규약
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path in ("/schedule", "/schedule.html"):
                return self._static("schedule.html")
            if path in ("/schedule_py", "/schedule_py.html"):
                return self._static("schedule_py.html")
            if path in ("/glossary", "/glossary.html"):
                return self._static("glossary.html")
            if path in ("/depgraph", "/depgraph.html"):
                return self._static("depgraph.html")
            if path == "/api/glossary":
                return self._json(200, glossary.build())
            if path == "/api/depgraph":
                # 저장소를 매번 파싱한다. 캐시하면 코드를 고쳐도 그림이 안 바뀐다.
                return self._json(200, depgraph.build())
            if path == "/api/anatomy":
                # ?module=schedule&func=apply  (func 를 빼면 모듈 목록)
                q = dict(
                    kv.split("=", 1)
                    for kv in (self.path.split("?", 1)[1].split("&") if "?" in self.path else [])
                    if "=" in kv
                )
                mod = q.get("module", "schedule")
                fn = q.get("func")
                try:
                    if fn:
                        return self._json(200, anatomy.describe(mod, fn))
                    return self._json(200, anatomy.module_index(mod))
                except KeyError as exc:
                    return self._fail(404, str(exc))
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/health":
                return self._json(200, {"ok": True, "backend": matmul.backend_name()})
            self._fail(404, f"없는 경로입니다: {path}")

        do_HEAD = do_GET

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                data = self._read_json()
            except (ValueError, json.JSONDecodeError) as exc:
                return self._fail(400, f"본문을 읽지 못했습니다: {exc}")

            try:
                if path == "/api/random":
                    m = int(data.get("m", DEFAULT_DIMS[0]))
                    n = int(data.get("n", DEFAULT_DIMS[1]))
                    p = int(data.get("p", DEFAULT_DIMS[2]))
                    seed = data.get("seed")
                    return self._json(
                        200, _random_state(m, n, p, None if seed is None else int(seed))
                    )

                if path == "/api/schedule":
                    return self._json(200, schedule_demo.run(data))

                if path == "/api/compute":
                    if "A" not in data or "B" not in data:
                        return self._fail(400, "A 와 B 가 모두 필요합니다")
                    return self._json(
                        200, build_state(data["A"], data["B"], data.get("i"), data.get("j"))
                    )

            except (ValueError, TypeError, IndexError) as exc:
                return self._fail(400, str(exc))
            except AssertionError as exc:  # 교차 검증 실패 — 조용히 넘기지 않는다
                return self._fail(500, f"검증 실패: {exc}")

            self._fail(404, f"없는 경로입니다: {path}")

        # -- 정적 파일 ------------------------------------------------------
        def _static(self, rel: str) -> None:
            target = (STATIC_DIR / rel).resolve()
            try:  # 경로 탈출 차단
                target.relative_to(STATIC_DIR)
            except ValueError:
                return self._fail(403, "허용되지 않은 경로입니다")
            if not target.is_file():
                return self._fail(404, f"파일이 없습니다: {rel}")

            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in (
                "application/javascript",
                "application/json",
            ):
                ctype += "; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

        # -- 로그 ------------------------------------------------------------
        def log_message(self, fmt: str, *args: Any) -> None:
            if not quiet:
                sys.stderr.write(
                    "[viz] %s - %s\n" % (self.address_string(), fmt % args)
                )

    return Handler


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def serve(host: str = "127.0.0.1", port: int = 8000, quiet: bool = False) -> ThreadingHTTPServer:
    """서버를 만들어 돌려준다 (``serve_forever`` 는 호출자 몫)."""
    return ThreadingHTTPServer((host, port), make_handler(quiet=quiet))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m utils.visualization_example",
        description="행렬 곱 시각화 데모 — 계산은 파이썬이, 그리기는 브라우저가 한다.",
    )
    ap.add_argument("--host", default="127.0.0.1", help="바인딩 주소 (기본: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="포트 (기본: 8000)")
    ap.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않는다")
    ap.add_argument("--quiet", action="store_true", help="접속 로그를 찍지 않는다")
    args = ap.parse_args(argv)

    if not STATIC_DIR.is_dir():
        print(f"[viz] 정적 파일 폴더가 없습니다: {STATIC_DIR}", file=sys.stderr)
        return 1

    try:
        httpd = serve(args.host, args.port, quiet=args.quiet)
    except OSError as exc:
        print(f"[viz] {args.host}:{args.port} 를 열지 못했습니다 — {exc}", file=sys.stderr)
        print("[viz] --port 로 다른 포트를 지정해 보세요.", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}/"
    print(f"[viz] 계산 백엔드: {matmul.backend_name()}")
    print(f"[viz] {url} 에서 실행 중 — 종료하려면 Ctrl+C")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz] 종료합니다.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
