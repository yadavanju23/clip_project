"""Search service for image similarity lookup."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from PIL import Image

from app.model import FeatureExtractor

LOGGER = logging.getLogger(__name__)


class ImageSearchService:
    """Load index and run top-k image similarity search."""

    def __init__(self, index_path: Path, paths_path: Path, extractor: FeatureExtractor) -> None:
        """Initialize service with index assets and model extractor."""
        self.index_path = index_path
        self.paths_path = paths_path
        self.extractor = extractor
        self._index: faiss.Index | None = None
        self._paths: list[str] | None = None

    def load(self) -> None:
        """Load FAISS index and path mapping from disk."""
        if not self.index_path.exists() or not self.paths_path.exists():
            raise FileNotFoundError(
                f"Index assets missing. Expected: {self.index_path} and {self.paths_path}"
            )

        self._index = faiss.read_index(str(self.index_path))
        with self.paths_path.open("rb") as f:
            self._paths = pickle.load(f)

        if not isinstance(self._paths, list):
            raise ValueError("paths.pkl format is invalid; expected list[str].")

        LOGGER.info("Loaded index with %d items", self._index.ntotal)

    def search(self, image: Image.Image, top_k: int = 20, threshold: float = 0.7) -> List[Dict[str, float]]:
        """Search for similar images and return filtered top-k results."""
        if self._index is None or self._paths is None:
            self.load()

        emb = self.extractor.extract(image)
        query = np.expand_dims(emb, axis=0).astype(np.float32)
        faiss.normalize_L2(query)

        top_k = max(1, top_k)
        candidate_k = min(max(top_k * 5, top_k), len(self._paths))
        distances, indices = self._index.search(query, candidate_k)

        results: List[Dict[str, float]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._paths):
                continue
            similarity = float(score)
            if similarity < threshold:
                continue
            results.append({"image_path": self._paths[idx], "similarity": round(similarity, 4)})
            if len(results) >= top_k:
                break

        return results
