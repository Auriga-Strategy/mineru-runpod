from __future__ import annotations

import json
from pathlib import Path

import pytest

import model_bootstrap


VLM_REVISION = "bff20d4ae2bf202df9f45284b4d43681555a97ed"
PIPELINE_REVISION = "ed6b654c018d742e65a17671e379c5e6ecc87ec9"


def _snapshot(root: Path, repo_id: str, revision: str, *, main_ref: str | None = None) -> Path:
    organization, name = repo_id.split("/", 1)
    model_root = root / f"models--{organization}--{name}"
    snapshot = model_root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    if main_ref is not None:
        (model_root / "refs").mkdir()
        (model_root / "refs" / "main").write_text(main_ref, encoding="utf-8")
    return snapshot


def test_configure_model_paths_requires_exact_cached_main_revision(
    monkeypatch, tmp_path
):
    baked = tmp_path / "baked"
    cached = tmp_path / "cached"
    _snapshot(baked, model_bootstrap.VLM_REPO, VLM_REVISION)
    _snapshot(
        cached,
        model_bootstrap.PIPELINE_REPO,
        PIPELINE_REVISION,
        main_ref="0" * 40,
    )
    monkeypatch.setenv("PGB_MINERU_VLM_REVISION", VLM_REVISION)
    monkeypatch.setenv("PGB_MINERU_PIPELINE_REVISION", PIPELINE_REVISION)
    with pytest.raises(RuntimeError, match="refs/main does not match"):
        model_bootstrap.configure_model_paths(
            baked_cache_root=baked,
            runpod_cache_root=cached,
            config_path=tmp_path / "mineru.json",
            mineru_validator=lambda _pipeline, _vlm: None,
        )


def test_configure_model_paths_writes_only_the_two_pinned_snapshots(
    monkeypatch, tmp_path
):
    baked = tmp_path / "baked"
    cached = tmp_path / "cached"
    vlm = _snapshot(baked, model_bootstrap.VLM_REPO, VLM_REVISION)
    pipeline = _snapshot(
        cached,
        model_bootstrap.PIPELINE_REPO,
        PIPELINE_REVISION,
        main_ref=PIPELINE_REVISION,
    )
    monkeypatch.setenv("PGB_MINERU_VLM_REVISION", VLM_REVISION)
    monkeypatch.setenv("PGB_MINERU_PIPELINE_REVISION", PIPELINE_REVISION)
    config_path = tmp_path / "mineru.json"
    validated = []
    result = model_bootstrap.configure_model_paths(
        baked_cache_root=baked,
        runpod_cache_root=cached,
        config_path=config_path,
        mineru_validator=lambda pipeline_path, vlm_path: validated.append(
            (pipeline_path, vlm_path)
        ),
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["models-dir"] == {"pipeline": str(pipeline), "vlm": str(vlm)}
    assert config["model-source"] == "local"
    assert validated == [(pipeline, vlm)]
    assert result == {
        "pipeline_revision": PIPELINE_REVISION,
        "vlm_revision": VLM_REVISION,
    }
