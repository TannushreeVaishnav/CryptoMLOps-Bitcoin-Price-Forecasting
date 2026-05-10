## MLflow Experiment Tracking

Full experiment lifecycle management with MLflow.

### Components
- `src/train.py` — Logs params, metrics, and model artifacts per run
- `mlflow_tracking.db` — Local SQLite tracking store (gitignored)
- Model Registry — Champion vs Challenger promotion logic
- `dvc.lock` — Reproducibility lock for each experiment

### Access MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow_tracking.db
# Open http://localhost:5000
```
