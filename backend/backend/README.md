# Diabetic Retinopathy Backend (FastAPI + Keras)

## Folder layout (relevant)
- `backend/backend/application.py` — FastAPI app (exports `app`)
- `backend/backend/model/` — model file(s), e.g. `best_ra_finetune_export.keras`
- `backend/backend/requirements.txt` — dependencies
- `backend/backend/server.py` — local runner (uvicorn)

## Local run (Windows / dev)

From `backend/backend`:

```bash
python server.py
```

Or directly with uvicorn:

```bash
uvicorn application:app --host 127.0.0.1 --port 5000 --reload
```

## Render deployment

### Root Directory
Set Render **Root Directory** to:

- `backend/backend`

(That folder must contain `requirements.txt` and `application.py`.)

### Build Command
```bash
pip install -r requirements.txt
```

### Start Command
```bash
sh -c "gunicorn -k uvicorn.workers.UvicornWorker -w 1 -t 600 -b 0.0.0.0:${PORT} application:app"

```

### Recommended Render env vars
- `ALLOWED_ORIGINS` = your Vercel production domain (comma-separated, no trailing slashes)
- `WEB_CONCURRENCY` = `1`
- `MODEL_FILE` = `best_ra_finetune_export.keras` (if you rename the file)

## Model path resolution
The backend searches for the model in:

1) `backend/backend/model/<MODEL_FILE>`
2) `backend/backend/<MODEL_FILE>`

You can confirm the resolved `model_path` via `GET /api/health`.

## API endpoints
- `GET /api/health`
- `POST /api/predict` (also available as `POST /predict`)
- `GET /api/model-info`
- `GET /api/privacy-notice`
- `GET /api/audit-log`
