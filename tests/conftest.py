"""pytest 가 저장소 루트를 임포트 경로에 넣게 한다 (설치 없이 실행 가능)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
