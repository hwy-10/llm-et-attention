"""§5 추가 할당 검증 — config · dataset · seeding  (팀 2)

이 세 파일은 "조용히 틀리는" 경로가 모여 있는 곳이다. 값이 이상해도 예외가 안 나고
기본값이 진짜 값과 같으면 아무도 모른다. PART1 이 실제로 그 사고를 당했다.
"""

import copy

import numpy as np
import pytest

from src.config import ConfigKeyError, _mini_yaml_load, load_config
from src.dataset import synthetic_qk


# ---------------------------------------------------------------------------
# config.py — 조용한 실패 경로
# ---------------------------------------------------------------------------
def test_missing_path_is_silent_by_default_but_loud_on_demand():
    """★ 없는 경로가 조용히 기본값을 내는 것이 실제 사고를 냈다.

    `quant.quant.n_planes`(존재하지 않음)가 기본값 8 로 통과했는데 진짜 값도 8 이라
    아무도 몰랐다. `required=True` 로 그 자리를 막을 수 있어야 한다.
    """
    cfg = load_config()
    assert cfg.get("quant.planes.n_planes") == 8          # 실제 경로
    assert cfg.get("quant.quant.n_planes", 8) == 8        # 없는 경로 — 조용히 기본값
    with pytest.raises(ConfigKeyError):
        cfg.get("quant.quant.n_planes", 8, required=True)
    with pytest.raises(ConfigKeyError):
        cfg.require("hardware.no_such_section.value")


def test_provenance_catches_misspelled_estimate():
    """★ `v == "estimate"` 정확 일치만 보면 오타가 전부 통과한다.

    RTL 팀이 값을 채우며 대문자나 공백을 남기면 **실측으로 교체된 것처럼 보인다.**
    """
    cfg = load_config()
    base = len(cfg.provenance_warnings())
    assert base > 0, "지금은 추정치가 있어야 한다"
    for typo in ("Estimate", "ESTIMATE", "estimated", "estimate "):
        c = copy.deepcopy(cfg)
        c.hardware["memory"]["source"] = typo
        assert len(c.provenance_warnings()) == base, f"{typo!r} 를 놓쳤다"


def test_tab_indentation_is_rejected_like_pyyaml_does():
    """★ 탭은 PyYAML 이 ScannerError 로 막는데 미니 파서는 조용히 통과시켰다.

    `a:\n\tb: 1` 이 `{'a': {}, 'b': 1}` 이 되어 **구조가 달라지는데 에러가 없다.**
    PyYAML 설치 여부에 따라 결과가 갈리면 안 된다.
    """
    with pytest.raises(ValueError):
        _mini_yaml_load("a:\n\tb: 1\n")
    # 공백 들여쓰기는 정상
    assert _mini_yaml_load("a:\n  b: 1\n") == {"a": {"b": 1}}


def test_mini_parser_matches_pyyaml_on_every_config_file():
    """PyYAML 이 있든 없든 같은 결과여야 한다."""
    try:
        import yaml
    except ImportError:
        return                      # PyYAML 이 없으면 비교할 대상이 없다
    from pathlib import Path

    from src.config import CONFIG_DIR, CONFIG_FILES

    for name in CONFIG_FILES:
        text = (Path(CONFIG_DIR) / f"{name}.yaml").read_text(encoding="utf-8")
        assert yaml.safe_load(text) == _mini_yaml_load(text), name


# ---------------------------------------------------------------------------
# dataset.py — 합성 데이터의 결정성
# ---------------------------------------------------------------------------
def test_sink_count_is_deterministic_across_seeds():
    """★ sink 개수가 시드마다 달라지던 자리.

    예전에는 `rng.choice(seq_len, ...)` 결과에 0 을 붙이고 `unique` 로 합쳤는데,
    choice 가 0 을 뽑으면 개수가 하나 줄었다 (200시드 중 10회).
    통제된 데이터 생성기가 시드에 따라 다른 개수를 내면 재현성 주장이 약해진다.
    """
    counts = {len(synthetic_qk(seq_len=512, head_dim=64, seed=s).meta["sink_idx"])
              for s in range(60)}
    assert len(counts) == 1, f"sink 개수가 시드마다 다르다: {sorted(counts)}"
    assert counts.pop() == 21, "0번 토큰 + round(0.04*512)=20 개"


def test_token_zero_is_always_a_sink():
    """실제 LLM 관측과 맞춘 성질 — 첫 토큰은 항상 sink."""
    for s in (0, 1, 7):
        assert 0 in synthetic_qk(seq_len=128, head_dim=64, seed=s).meta["sink_idx"]


def test_same_seed_gives_identical_tensors():
    """같은 시드로 여러 사람이 같은 결과를 얻어야 한다."""
    a = synthetic_qk(seq_len=64, head_dim=32, seed=3)
    b = synthetic_qk(seq_len=64, head_dim=32, seed=3)
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.k, b.k)
    c = synthetic_qk(seq_len=64, head_dim=32, seed=4)
    assert not np.array_equal(a.q, c.q), "다른 시드는 달라야 한다"


# ---------------------------------------------------------------------------
# seeding.py — 아날로그 잔재 제거 확인
# ---------------------------------------------------------------------------
def test_seeding_exposes_only_what_is_used():
    """★ 이전 판에는 아날로그 CIM 전제의 instance_rng / trial_rng 가 있었다.

    (매크로 인스턴스 INL, 커패시터 미스매치, 열잡음 몬테카를로) 이 프로젝트는
    디지털만 다루므로 범위 밖이고, 저장소 어디서도 호출되지 않는 죽은 코드였다.
    """
    import src.seeding as seeding

    assert hasattr(seeding, "data_rng")
    for gone in ("instance_rng", "trial_rng"):
        assert not hasattr(seeding, gone), f"{gone} 이 되살아났다 — 아날로그는 범위 밖이다"
    # 독스트링에는 '왜 지웠는지'가 남아 있어도 된다. 함수가 없으면 된다.
    assert "data_rng" in seeding.__all__ if hasattr(seeding, "__all__") else True


def test_cache_fallback_does_not_cross_layers_or_heads():
    """★ 폴백이 다른 층·헤드의 텐서를 집어 오던 자리.

    요청한 길이보다 짧은 캐시밖에 없을 때 `cache.parent.glob("*.npz")` 로
    **아무 npz 나** 가져왔다. `head_dim` 만 맞으면 통과해서, 8층 0번 헤드를
    요청했는데 **2층 5번 헤드 텐서로 실험이 도는** 사고가 가능했다.

    실제 캐시와 섞이지 않도록 모델 이름을 바꿔 전용 파일만 두고 확인한다.
    """
    import numpy as np

    from src.dataset import QKSnapshot, cached_snapshot_path, snapshot_from_config

    cfg = load_config()
    cfg.model["model"]["name"] = "PYTEST-FALLBACK"      # 실제 캐시와 이름 분리
    d = int(cfg.head_dim)
    want = cached_snapshot_path(cfg, 400)
    stem = want.stem.rsplit("_T", 1)[0]                 # PYTEST-FALLBACK_L8_H0

    short_same = want.with_name(f"{stem}_T64.npz")                       # 같은 헤드, 너무 짧음
    long_other = want.with_name(stem.replace("_L8_H0", "_L2_H5") + "_T512.npz")  # 다른 헤드, 충분히 김
    want.parent.mkdir(parents=True, exist_ok=True)
    try:
        QKSnapshot(q=np.zeros((64, d)), k=np.zeros((64, d)), source="short-same").save(short_same)
        QKSnapshot(q=np.ones((512, d)), k=np.ones((512, d)), source="long-other").save(long_other)

        snap = snapshot_from_config(cfg, seq_len=400)

        # 다른 층·헤드(long-other)를 절대 쓰면 안 된다. 쓸 게 없으니 합성으로 떨어져야 한다.
        assert snap.source != "long-other", (
            "폴백이 다른 층·헤드(L2 H5)의 텐서를 집어 왔다"
        )
        assert snap.source == "synthetic"
    finally:
        short_same.unlink(missing_ok=True)
        long_other.unlink(missing_ok=True)
