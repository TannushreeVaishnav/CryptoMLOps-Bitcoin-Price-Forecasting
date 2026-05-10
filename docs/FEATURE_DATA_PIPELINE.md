## Data Pipeline

End-to-end data ingestion and preprocessing pipeline powered by DVC.

### Components
- `src/data/loader.py` — Fetches live BTC OHLCV via Yahoo Finance
- `src/data/preprocessor.py` — RSI, EMA, MACD, Bollinger Bands feature engineering
- `src/data/splitter.py` — Time-series aware train/val/test split
- `dvc.yaml` — Pipeline stages: ingest → preprocess → split
- `params.yaml` — All configurable hyperparameters

### Run
```bash
dvc repro
```
