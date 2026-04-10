"""Tools for building and persisting FAISS indexes."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.model import FeatureExtractor

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class IndexBuilder:
    """Build FAISS index from an image directory."""

    def __init__(self, extractor: FeatureExtractor) -> None:
        """Initialize the index builder with a feature extractor."""
        self.extractor = extractor

    def collect_images(self, images_dir: Path) -> List[Path]:
        """Collect and sort all supported image files recursively."""
        all_paths = [
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(all_paths)

    def build_embeddings(self, image_paths: Sequence[Path]) -> tuple[np.ndarray, list[Path]]:
        """Create normalized embeddings for all provided images."""
        vectors = []
        valid_paths = []

        for path in image_paths:
            try:
                with Image.open(path) as img:
                    emb = self.extractor.extract(img)
                vectors.append(emb)
                valid_paths.append(path)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                LOGGER.warning("Skipping invalid image %s: %s", path, exc)

        if not vectors:
            raise ValueError("No valid images found for indexing.")

        embeddings = np.vstack(vectors).astype(np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings, valid_paths

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create cosine-similarity FAISS index (Inner Product on normalized vectors)."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def save(self, index: faiss.Index, paths: Sequence[Path], index_path: Path, paths_path: Path) -> None:
        """Persist FAISS index and image path mapping to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        paths_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(index_path))
        with paths_path.open("wb") as f:
            pickle.dump([str(p) for p in paths], f)

        LOGGER.info("Saved index to %s", index_path)
        LOGGER.info("Saved paths to %s", paths_path)

    def build_and_save(self, images_dir: Path, index_path: Path, paths_path: Path) -> int:
        """End-to-end index build and persistence. Returns indexed image count."""
        image_paths = self.collect_images(images_dir)
        if not image_paths:
            raise ValueError(f"No images found in {images_dir}")

        embeddings, valid_paths = self.build_embeddings(image_paths)
        index = self.build_faiss_index(embeddings)
        self.save(index, valid_paths, index_path, paths_path)
        return len(valid_paths)
