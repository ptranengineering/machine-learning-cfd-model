# Surrogate design optimizer (web UI)

Interactive **sliders and inputs** for the search box, plus **CL / CD** surrogate optimization (same logic as `ml/optimize_design.py`).

## Why not GitHub Pages alone?

**GitHub Pages** only serves static files. The sklearn **joblib** model must run in **Python**, so this app uses **FastAPI** (API + static assets). You still deploy **from GitHub** using free hosts that run Python (below).

## Local run

From the **repository root** (where `ml/` and `results/` live):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r webapp/requirements.txt
```

Ensure the trained model exists:

`results/models/design_rf_model.joblib`

Then:

```bash
uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**

### Custom model path

```bash
export MODEL_PATH=/path/to/design_rf_model.joblib
uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

## Deploy from GitHub (example: Render)

1. Push this repo to GitHub.
2. New **Web Service** → connect the repo.
3. **Build command:** `pip install -r webapp/requirements.txt`
4. **Start command:** `uvicorn webapp.main:app --host 0.0.0.0 --port $PORT`
5. **Root directory:** leave default (repo root) so `ml/` imports work.
6. Upload or generate `design_rf_model.joblib` on the host, or set **environment variable** `MODEL_PATH` to the file location.

If the service sleeps on free tier, first load may take a few seconds.

## Docker (optional)

```bash
docker build -t aero-opt-web -f webapp/Dockerfile .
docker run -p 8000:8000 -v "$(pwd)/results/models:/models:ro" \
  -e MODEL_PATH=/models/design_rf_model.joblib aero-opt-web
```

Requires `webapp/Dockerfile` (see repo).
