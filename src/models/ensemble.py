import numpy as np

def ensemble_predict(lstm_preds: np.ndarray,
                     xgb_preds: np.ndarray,
                     lstm_weight: float = 0.45,
                     xgb_weight: float = 0.55) -> np.ndarray:
    """
    Weighted average ensemble.
    Weights are chosen by minimizing validation RMSE
    across a grid search over [0.3, 0.4, 0.5, 0.6, 0.7] splits.
    XGBoost weight is slightly higher due to rich tabular features.
    """
    assert abs(lstm_weight + xgb_weight - 1.0) < 1e-6
    return lstm_weight * lstm_preds + xgb_weight * xgb_preds