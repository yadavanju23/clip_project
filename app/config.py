"""Configuration helpers for the image similarity search system."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application runtime settings loaded from environment variables."""

    project_root: Path
    data_dir: Path
    images_dir: Path
    index_path: Path
    paths_path: Path
    temp_upload_dir: Path
    model_name: str
    default_top_k: int
    default_threshold: float
    backend_host: str
    backend_port: int


def _to_abs(project_root: Path, path_value: str) -> Path:
    """Resolve a path to an absolute path against the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else (project_root / path).resolve()


def get_settings() -> Settings:
    """Build settings object from environment variables with safe defaults."""
    project_root = Path(__file__).resolve().parents[1]

    data_dir = _to_abs(project_root, os.getenv("DATA_DIR", "data"))
    images_dir = _to_abs(project_root, os.getenv("IMAGES_DIR", str(data_dir / "images")))
    index_path = _to_abs(project_root, os.getenv("INDEX_PATH", str(data_dir / "index.faiss")))
    paths_path = _to_abs(project_root, os.getenv("PATHS_PATH", str(data_dir / "paths.pkl")))
    temp_upload_dir = _to_abs(project_root, os.getenv("TEMP_UPLOAD_DIR", str(data_dir / "tmp_uploads")))

    model_name = os.getenv("MODEL_NAME", "resnet50")
    default_top_k = int(os.getenv("DEFAULT_TOP_K", "20"))
    default_threshold = float(os.getenv("DEFAULT_THRESHOLD", "0.7"))
    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port = int(os.getenv("BACKEND_PORT", "8000"))

    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        images_dir=images_dir,
        index_path=index_path,
        paths_path=paths_path,
        temp_upload_dir=temp_upload_dir,
        model_name=model_name,
        default_top_k=default_top_k,
        default_threshold=default_threshold,
        backend_host=backend_host,
        backend_port=backend_port,
    )
