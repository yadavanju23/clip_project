"""CLI script to build FAISS image index."""

from __future__ import annotations

import logging

from app.config import get_settings
from app.indexer import IndexBuilder
from app.model import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Build image embeddings index from configured image directory."""
    settings = get_settings()
    settings.images_dir.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor(model_name=settings.model_name)
    builder = IndexBuilder(extractor=extractor)

    count = builder.build_and_save(
        images_dir=settings.images_dir,
        index_path=settings.index_path,
        paths_path=settings.paths_path,
    )
    LOGGER.info("Index built successfully with %d images", count)


if __name__ == "__main__":
    main()
