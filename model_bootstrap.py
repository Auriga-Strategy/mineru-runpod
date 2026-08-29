"""Fail-closed MinerU model configuration for the PGB RunPod worker."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable


VLM_REPO = "opendatalab/MinerU2.5-Pro-2605-1.2B"
PIPELINE_REPO = "opendatalab/PDF-Extract-Kit-1.0"
BAKED_CACHE_ROOT = Path("/root/.cache/huggingface/hub")
RUNPOD_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
DEFAULT_CONFIG_PATH = Path("/root/mineru.json")


def _required_revision(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{name} must be an immutable lowercase commit hash")
    return value


def _model_root(cache_root: Path, repo_id: str) -> Path:
    organization, name = repo_id.split("/", 1)
    return cache_root / f"models--{organization}--{name}"


def _exact_snapshot(
    cache_root: Path,
    repo_id: str,
    revision: str,
    *,
    require_main_ref: bool,
) -> Path:
    root = _model_root(cache_root, repo_id)
    snapshot = root / "snapshots" / revision
    if require_main_ref:
        ref = root / "refs" / "main"
        if not ref.is_file() or ref.read_text(encoding="utf-8").strip() != revision:
            raise RuntimeError(f"cached model refs/main does not match pinned {repo_id} revision")
    if not snapshot.is_dir():
        raise RuntimeError(f"pinned {repo_id} snapshot is unavailable")
    return snapshot


def _validate_with_mineru(pipeline: Path, vlm: Path) -> None:
    from mineru.utils.enum_class import ModelPath
    from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

    resolved_pipeline = Path(
        auto_download_and_get_model_root_path(ModelPath.pp_doclayout_v2, repo_mode="pipeline")
    )
    resolved_vlm = Path(auto_download_and_get_model_root_path("/", repo_mode="vlm"))
    required = (
        ModelPath.pp_doclayout_v2,
        ModelPath.unimernet_small,
        ModelPath.pp_formulanet_plus_m,
        ModelPath.pytorch_paddle,
        ModelPath.slanet_plus,
        ModelPath.unet_structure,
        ModelPath.paddle_table_cls,
    )
    if resolved_pipeline != pipeline or resolved_vlm != vlm:
        raise RuntimeError("MinerU local model resolver returned an unpinned path")
    if not all((pipeline / item).exists() for item in required):
        raise RuntimeError("pinned pipeline snapshot is incomplete")
    if not (vlm / "config.json").is_file():
        raise RuntimeError("pinned VLM snapshot is incomplete")


def configure_model_paths(
    *,
    baked_cache_root: Path = BAKED_CACHE_ROOT,
    runpod_cache_root: Path = RUNPOD_CACHE_ROOT,
    config_path: Path | None = None,
    mineru_validator: Callable[[Path, Path], None] = _validate_with_mineru,
) -> dict[str, str]:
    vlm_revision = _required_revision("PGB_MINERU_VLM_REVISION")
    pipeline_revision = _required_revision("PGB_MINERU_PIPELINE_REVISION")
    vlm = _exact_snapshot(
        baked_cache_root, VLM_REPO, vlm_revision, require_main_ref=False
    )
    pipeline = _exact_snapshot(
        runpod_cache_root, PIPELINE_REPO, pipeline_revision, require_main_ref=True
    )
    target = config_path or Path(
        os.environ.get("MINERU_TOOLS_CONFIG_JSON", str(DEFAULT_CONFIG_PATH))
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "models-dir": {"pipeline": str(pipeline), "vlm": str(vlm)},
        "model-source": "local",
        "config_version": "1.3.2",
    }
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)
    mineru_validator(pipeline, vlm)
    return {
        "pipeline_revision": pipeline_revision,
        "vlm_revision": vlm_revision,
    }


def main() -> None:
    revisions = configure_model_paths()
    print(
        json.dumps(
            {"event": "pinned_models_ready", **revisions},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    handler = Path(__file__).with_name("handler.py")
    os.execv(sys.executable, [sys.executable, "-u", str(handler)])


if __name__ == "__main__":
    main()
