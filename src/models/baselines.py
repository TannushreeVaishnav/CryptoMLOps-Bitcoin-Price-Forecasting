import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import logging

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

def compute_metrics(y_true, y_pred, y_prev):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    dir_acc = np.mean(true_dir == pred_dir) * 100
    return mae, rmse, mape, dir_acc

def evaluate_baselines(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                       target: str, seq_len: int, scaler) -> dict:
    print("Training Baselines (Linear Regression & Prophet)...")
    
    # ---------------------------------------------------------
    # 1. Linear Regression (Tabular)
    # ---------------------------------------------------------
    y_train_lr = train_df[target].shift(-1).dropna()
    X_train_lr = train_df.iloc[:-1]
    
    y_test_lr = test_df[target].shift(-1).dropna()
    X_test_lr = test_df.iloc[:-1]
    
    lr = LinearRegression()
    lr.fit(X_train_lr, y_train_lr)
    lr_preds = lr.predict(X_test_lr)
    
    # ---------------------------------------------------------
    # 2. Facebook Prophet (Time Series)
    # ---------------------------------------------------------
    # Prophet requires raw, unscaled prices, so we use the unscaled train data
    # We will grab unscaled prices using the scaler logic from train_flow
    close_idx = list(train_df.columns).index('Close')
    close_mean = scaler.mean_[close_idx]
    close_scale = scaler.scale_[close_idx]
    
    def unscale_close(x): return x * close_scale + close_mean
    
    # Format for Prophet: ds and y
    train_prices = unscale_close(train_df['Close'].values)
    prophet_train = pd.DataFrame({
        'ds': train_df.index,
        'y': train_prices
    })
    
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(prophet_train)
    
    # Predict on test set dates
    future = pd.DataFrame({'ds': test_df.index})
    prophet_forecast = m.predict(future)
    prophet_preds_close = prophet_forecast['yhat'].values
    
    # ---------------------------------------------------------
    # Metrics Calculation
    # ---------------------------------------------------------
    # Align lengths (test_df is chopped by seq_len in train_flow evaluate_ensemble)
    y_test_true_close = unscale_close(test_df['Close'].values[seq_len:])
    y_test_prev_close = unscale_close(test_df['Close'].values[seq_len-1:-1])
    
    # LR unscaling (predicts daily return)
    target_idx = list(train_df.columns).index(target)
    target_mean = scaler.mean_[target_idx]
    target_scale = scaler.scale_[target_idx]
    def unscale_return(x): return x * target_scale + target_mean
    
    lr_preds_aligned = lr_preds[-len(y_test_true_close):]
    lr_preds_ret = unscale_return(lr_preds_aligned)
    lr_preds_close = y_test_prev_close * (1 + lr_preds_ret)
    
    # Prophet alignment
    prophet_preds_aligned = prophet_preds_close[-len(y_test_true_close):]
    
    # Compute metrics
    lr_mae, lr_rmse, lr_mape, lr_dir_acc = compute_metrics(y_test_true_close, lr_preds_close, y_test_prev_close)
    pr_mae, pr_rmse, pr_mape, pr_dir_acc = compute_metrics(y_test_true_close, prophet_preds_aligned, y_test_prev_close)
    
    return {
        "lr_mae": float(lr_mae),
        "lr_rmse": float(lr_rmse),
        "lr_mape": float(lr_mape),
        "lr_dir_acc": float(lr_dir_acc),
        "prophet_mae": float(pr_mae),
        "prophet_rmse": float(pr_rmse),
        "prophet_mape": float(pr_mape),
        "prophet_dir_acc": float(pr_dir_acc)
    }
