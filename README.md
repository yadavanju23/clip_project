# Image Similarity Search System

## Project Overview
This project is an internal reverse-image-search system for product catalogs. A user uploads a query image, the backend extracts an embedding using a pretrained deep model, and FAISS returns the most visually similar images from an indexed internal dataset.

## Features
- ResNet50 pretrained feature extraction (last layer removed)
- Optional CLIP visual backbone (`MODEL_NAME=clip`)
- FAISS cosine similarity search (`IndexFlatIP` + L2 normalization)
- FastAPI backend with file upload, validation, and CORS
- Static image serving via `/images/*` for browser-safe result rendering
- Modern frontend with preview, drag-and-drop, threshold slider, loading state, and responsive grid
- Config-driven paths and runtime settings via environment variables
- Automated index build script
- Basic unit test for search pipeline

## Demo Flow (Upload -> Results)
1. Add dataset images to `data/images/`
2. Build FAISS index using `scripts/build_index.py`
3. Start FastAPI backend
4. Open frontend page and upload a query image
5. Review top similar results with similarity score overlays

## Tech Stack
- Python, FastAPI, Uvicorn
- PyTorch + Torchvision (ResNet50 / optional CLIP)
- FAISS (CPU)
- HTML, CSS, JavaScript (vanilla frontend)
- Pytest

## Installation Steps
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

## How to Build Index
```bash
python -m scripts.build_index
```

## How to Run Backend
```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8010
```

## How to Run Frontend
Serve static files from project root so image paths resolve:

```bash
python -m http.server 5500
```

Then open:
- Frontend UI: [http://127.0.0.1:5500/frontend/index.html](http://127.0.0.1:5500/frontend/index.html)
- API docs: [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs)

## Quick Start (Recommended)
Use separate terminals so both backend and frontend stay active.

1. Terminal A (backend):
```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8010
```

2. Terminal B (frontend static server):
```bash
python -m http.server 5500
```

3. Optional Terminal C (after adding/changing images):
```bash
python -m scripts.build_index
```

4. Open [http://127.0.0.1:5500/frontend/index.html](http://127.0.0.1:5500/frontend/index.html), upload an image, and search.

## API Usage
### Health
```bash
curl http://127.0.0.1:8010/health
```

### Search
```bash
curl -X POST "http://127.0.0.1:8010/search?top_k=20&threshold=0.7" \\
  -H "accept: application/json" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@data/images/sample.jpg"
```

Example response:
```json
{
  "results": [
    {
      "image_path": "data/images/xyz.jpg",
      "image_url": "http://127.0.0.1:8010/images/xyz.jpg",
      "similarity": 0.82
    }
  ]
}
```

## Troubleshooting
- `Failed to fetch`: backend is not running or wrong port is used. Confirm `http://127.0.0.1:8010/health`.
- Images not visible in result cards: restart backend and hard refresh browser (`Ctrl+F5`).
- Too few results: lower threshold (e.g. `0.4` to `0.6`) and rebuild index.
- New dataset images not appearing: run `python -m scripts.build_index` again.
- Port conflict on `8010`: run backend on another port and update `frontend/script.js` API URL.

## Folder Structure
```text
image-search/
|-- app/
|   |-- __init__.py
|   |-- model.py
|   |-- indexer.py
|   |-- search.py
|   |-- config.py
|-- api/
|   |-- main.py
|-- frontend/
|   |-- index.html
|   |-- style.css
|   |-- script.js
|-- data/
|   |-- images/
|   |-- index.faiss
|   |-- paths.pkl
|-- scripts/
|   |-- build_index.py
|-- tests/
|   |-- test_search.py
|-- requirements.txt
|-- README.md
|-- .env.example
|-- .gitignore
```

## Screenshots
- Add screenshots here:
  - Query upload screen
  - Results grid with similarity overlays

## Future Improvements
- Add ANN index choices (IVF, HNSW, PQ) for larger catalogs
- Add metadata filtering (category, brand)
- Persist query history and analytics
- Add authentication and role-based access
- Containerize deployment with Docker and CI/CD
