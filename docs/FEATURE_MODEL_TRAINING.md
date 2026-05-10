## Model Training

Ensemble model combining LSTM (sequence learning) and XGBoost (tabular boosting).

### Components
- `src/models/lstm.py` — PyTorch Lightning LSTM with configurable layers
- `src/models/xgboost_model.py` — XGBoost regressor with hyperparameter tuning
- `src/models/ensemble.py` — Weighted blending of LSTM + XGBoost predictions
- `src/train.py` — Prefect flow orchestrating the full training loop
- `params.yaml` — Learning rate, hidden_dim, n_estimators, blend_weight

### Run
```bash
python src/train.py
```
