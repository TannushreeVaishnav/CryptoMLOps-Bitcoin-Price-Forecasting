## API Serving

FastAPI-based REST inference API for real-time Bitcoin price predictions.

### Components
- `src/api/main.py` — FastAPI app with /predict endpoint
- `src/api/schema.py` — Pydantic request/response models
- Served via Uvicorn with automatic Swagger docs at `/docs`

### Run
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000/docs
```
