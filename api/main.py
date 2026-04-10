"""FastAPI application for image similarity search."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.model import FeatureExtractor
from app.search import ImageSearchService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOGGER = logging.getLogger(__name__)

settings = get_settings()
settings.temp_upload_dir.mkdir(parents=True, exist_ok=True)

extractor = FeatureExtractor(model_name=settings.model_name)
search_service = ImageSearchService(
    index_path=settings.index_path,
    paths_path=settings.paths_path,
    extractor=extractor,
)

app = FastAPI(title="Image Similarity Search API", version="1.0.0")
app.mount("/images", StaticFiles(directory=str(settings.images_dir)), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "OK"}


@app.post("/search")
async def search_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(default=settings.default_top_k, ge=1, le=100),
    threshold: float = Query(default=settings.default_threshold, ge=0.0, le=1.0),
) -> dict:
    """Accept uploaded image and return similar images from FAISS index."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    temp_path = settings.temp_upload_dir / f"query_{uuid.uuid4().hex}{suffix}"

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        temp_path.write_bytes(content)

        with Image.open(temp_path) as img:
            image = img.convert("RGB")
            results = search_service.search(image=image, top_k=top_k, threshold=threshold)

        serialized_results = []
        base_url = str(request.base_url).rstrip("/")
        for item in results:
            raw_path = Path(item["image_path"]).resolve()
            try:
                relative_path = raw_path.relative_to(settings.images_dir.resolve()).as_posix()
            except ValueError:
                LOGGER.warning("Skipping out-of-dataset path in result: %s", raw_path)
                continue

            serialized_results.append(
                {
                    "image_path": f"data/images/{relative_path}",
                    "image_url": f"{base_url}/images/{quote(relative_path, safe='/')}",
                    "similarity": item["similarity"],
                }
            )

        return {"results": serialized_results}
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not parse image file.")
    except FileNotFoundError as exc:
        LOGGER.exception("Index not found")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        LOGGER.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError as exc:
            LOGGER.warning("Failed to cleanup temp file %s: %s", temp_path, exc)
