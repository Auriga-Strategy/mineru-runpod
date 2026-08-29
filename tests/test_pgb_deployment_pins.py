"""PGB deployment reproducibility gates for the RunPod benchmark image."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mineru_and_top_level_runtime_dependencies_are_exactly_pinned():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "mineru[core,vllm]==3.4.5" in requirements
    assert "runpod==1.10.0" in requirements
    assert "httpx==0.28.1" in requirements
    assert "mineru[core,vllm]>=" not in requirements


def test_both_model_snapshots_use_immutable_revisions():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "MINERU_VLM_REVISION=bff20d4ae2bf202df9f45284b4d43681555a97ed" in dockerfile
    assert "MINERU_PIPELINE_REVISION=ed6b654c018d742e65a17671e379c5e6ecc87ec9" in dockerfile
    assert dockerfile.count("revision='${MINERU_") == 2
