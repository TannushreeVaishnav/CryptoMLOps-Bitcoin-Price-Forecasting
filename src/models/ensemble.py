import numpy as np

def find_best_weights(lstm_val_preds, xgb_val_preds, y_val):
    """
    Grid search over LSTM weight in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7].
    Choose weight pair that minimises validation RMSE.
    Log winning weights to MLflow.
    """
    best_rmse, best_w = float('inf'), 0.45
    for w in np.arange(0.3, 0.71, 0.05):
        preds = w * lstm_val_preds + (1 - w) * xgb_val_preds
        rmse  = np.sqrt(np.mean((preds - y_val) ** 2))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w
    return round(best_w, 2), round(1 - best_w, 2)

def ensemble_predict(lstm_preds, xgb_preds,
                     lstm_weight=0.45, xgb_weight=0.55):
    assert abs(lstm_weight + xgb_weight - 1.0) < 1e-6
    return lstm_weight * lstm_preds + xgb_weight * xgb_preds