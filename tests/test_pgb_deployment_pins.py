"""PGB deployment reproducibility gates for the RunPod benchmark image."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mineru_and_top_level_runtime_dependencies_are_exactly_pinned():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "mineru[core,vllm]==3.4.5" in requirements
    assert "runpod==1.10.0" in requirements
    assert "httpx==0.28.1" in requirements
    assert "mineru[core,vllm]>=" not in requirements


def test_baked_and_cached_model_snapshots_use_immutable_revisions():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "MINERU_VLM_REVISION=bff20d4ae2bf202df9f45284b4d43681555a97ed" in dockerfile
    assert "MINERU_PIPELINE_REVISION=ed6b654c018d742e65a17671e379c5e6ecc87ec9" in dockerfile
    assert dockerfile.count("revision='${MINERU_") == 1
    assert "snapshot_download(repo_id='opendatalab/PDF-Extract-Kit-1.0'" not in dockerfile
    assert "COPY model_bootstrap.py /worker/model_bootstrap.py" in dockerfile
    assert 'CMD ["python3", "-u", "model_bootstrap.py"]' in dockerfile


def test_ci_blocks_deployment_on_exact_linux_dependency_stage():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "AS dependency-preflight" in dockerfile
    assert "version('mineru') == '3.4.5'" in dockerfile
    assert "version('runpod') == '1.10.0'" in dockerfile
    assert "--python-platform x86_64-unknown-linux-gnu" in workflow
    assert "--target dependency-preflight" in workflow


def test_runtime_uses_minerus_documented_local_model_mode():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    bootstrap = (ROOT / "model_bootstrap.py").read_text(encoding="utf-8")
    assert "MINERU_MODEL_SOURCE=local" in dockerfile
    assert "MINERU_TOOLS_CONFIG_JSON=/root/mineru.json" in dockerfile
    assert '"model-source": "local"' in bootstrap
    assert "RUN --network=none python3 -c" in dockerfile
    assert "auto_download_and_get_model_root_path" in bootstrap
    assert "cached model refs/main does not match pinned" in bootstrap
