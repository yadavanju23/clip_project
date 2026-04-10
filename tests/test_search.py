"""Tests for image search service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.indexer import IndexBuilder
from app.search import ImageSearchService


class DummyExtractor:
    """Simple deterministic extractor for test isolation."""

    def extract(self, image: Image.Image) -> np.ndarray:
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        mean_val = float(arr.mean()) / 255.0
        return np.array([mean_val, 1.0 - mean_val], dtype=np.float32)


def _make_image(path: Path, value: int) -> None:
    """Create a small synthetic image for tests."""
    img = Image.new("RGB", (16, 16), color=(value, value, value))
    img.save(path)


def test_search_returns_results(tmp_path: Path) -> None:
    """Ensure search returns at least one similar image."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)

    _make_image(images_dir / "a.png", 30)
    _make_image(images_dir / "b.png", 200)

    extractor = DummyExtractor()
    builder = IndexBuilder(extractor=extractor)

    index_path = tmp_path / "index.faiss"
    paths_path = tmp_path / "paths.pkl"
    count = builder.build_and_save(images_dir, index_path, paths_path)

    assert count == 2

    service = ImageSearchService(index_path=index_path, paths_path=paths_path, extractor=extractor)
    query = Image.new("RGB", (16, 16), color=(35, 35, 35))

    results = service.search(query, top_k=5, threshold=0.0)

    assert len(results) >= 1
    assert "image_path" in results[0]
    assert "similarity" in results[0]
