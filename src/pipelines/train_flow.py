import os
import sys
import io

# Force UTF-8 encoding for stdout to handle MLflow emojis on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
import yaml
import mlflow
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from mlflow.tracking.client import MlflowClient

from src.models.xgb_model import train_xgboost as do_train_xgboost
from src.models.lstm_model import BTCLSTMModel
from src.models.lstm_dataset import BTCDataset
from src.models.ensemble import ensemble_predict
from prefect import flow, task
from dotenv import load_dotenv

load_dotenv()

def compute_metrics(y_true, y_pred, y_prev):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    dir_acc = np.mean(true_dir == pred_dir) * 100
    return mae, rmse, mape, dir_acc

@task(name="train_xgboost")
def train_xgboost(train_df, val_df, test_df, target, p_xgb):
    print("Training XGBoost...")
    y_train_xgb = train_df[target].shift(-1).dropna()
    X_train_xgb = train_df.iloc[:-1]
    y_val_xgb = val_df[target].shift(-1).dropna()
    X_val_xgb = val_df.iloc[:-1]
    y_test_xgb = test_df[target].shift(-1).dropna()
    X_test_xgb = test_df.iloc[:-1]
    
    xgb_model, shap_values = do_train_xgboost(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb, p_xgb)
    xgb_val_preds = xgb_model.predict(X_val_xgb)
    xgb_test_preds = xgb_model.predict(X_test_xgb)
    return xgb_model, xgb_val_preds, xgb_test_preds

@task(name="train_lstm")
def train_lstm(train_df, val_df, test_df, target, p_lstm):
    print("Training LSTM...")
    seq_len = p_lstm["sequence_length"]
    train_ds = BTCDataset(train_df, seq_len, target)
    val_ds = BTCDataset(val_df, seq_len, target)
    test_ds = BTCDataset(test_df, seq_len, target)
    
    train_dl = DataLoader(train_ds, batch_size=p_lstm["batch_size"], shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=p_lstm["batch_size"], shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=p_lstm["batch_size"], shuffle=False, num_workers=0)
    
    lstm_model = BTCLSTMModel(
        input_size=train_df.shape[1],
        hidden_size=p_lstm["hidden_size"],
        num_layers=p_lstm["num_layers"],
        dropout=p_lstm["dropout"],
        lr=p_lstm["learning_rate"],
        weight_decay=p_lstm.get("weight_decay", 1e-4)
    )
    
    trainer = pl.Trainer(max_epochs=p_lstm["max_epochs"], accelerator='cpu', devices=1)
    trainer.fit(lstm_model, train_dl, val_dl)
    trainer.save_checkpoint("artifacts/lstm_model.ckpt")
    
    lstm_model.eval()
    with torch.no_grad():
        lstm_val_preds = np.concatenate([lstm_model(x).numpy() for x, _ in val_dl])
        lstm_test_preds = np.concatenate([lstm_model(x).numpy() for x, _ in test_dl])
    return lstm_model, lstm_val_preds, lstm_test_preds

@task(name="evaluate_ensemble")
def evaluate_ensemble(train_df, val_df, test_df, target, p_lstm, p_ens, lstm_val_preds, lstm_test_preds, xgb_val_preds, xgb_test_preds):
    print("Evaluating Ensemble...")
    seq_len = p_lstm["sequence_length"]
    N_val = len(lstm_val_preds)
    N_test = len(lstm_test_preds)
    xgb_val_aligned = xgb_val_preds[-N_val:]
    xgb_test_aligned = xgb_test_preds[-N_test:]
    
    scaler = joblib.load("artifacts/scaler.pkl")
    target_idx = list(train_df.columns).index(target)
    target_mean = scaler.mean_[target_idx]
    target_scale = scaler.scale_[target_idx]
    close_idx = list(train_df.columns).index('Close')
    close_mean = scaler.mean_[close_idx]
    close_scale = scaler.scale_[close_idx]
    
    def unscale_return(x): return x * target_scale + target_mean
    def unscale_close(x): return x * close_scale + close_mean
    
    y_val_true_close = unscale_close(val_df['Close'].values[seq_len:])
    y_test_true_close = unscale_close(test_df['Close'].values[seq_len:])
    y_val_prev_close = unscale_close(val_df['Close'].values[seq_len-1:-1])
    y_test_prev_close = unscale_close(test_df['Close'].values[seq_len-1:-1])
    
    val_preds_ret = unscale_return(ensemble_predict(lstm_val_preds, xgb_val_aligned, p_ens["lstm_weight"], p_ens["xgb_weight"]))
    test_preds_ret = unscale_return(ensemble_predict(lstm_test_preds, xgb_test_aligned, p_ens["lstm_weight"], p_ens["xgb_weight"]))
    
    val_preds_close = y_val_prev_close * (1 + val_preds_ret)
    test_preds_close = y_test_prev_close * (1 + test_preds_ret)
    
    naive_mae, naive_rmse, naive_mape, _ = compute_metrics(y_test_true_close, y_test_prev_close, y_test_prev_close)
    val_mae, val_rmse, _, _ = compute_metrics(y_val_true_close, val_preds_close, y_val_prev_close)
    test_mae, test_rmse, test_mape, dir_acc = compute_metrics(y_test_true_close, test_preds_close, y_test_prev_close)
    
    # Individual isolated model predictions
    xgb_preds_close = y_test_prev_close * (1 + unscale_return(xgb_test_aligned))
    lstm_preds_close = y_test_prev_close * (1 + unscale_return(lstm_test_preds))
    
    xgb_mae, xgb_rmse, xgb_mape, xgb_dir = compute_metrics(y_test_true_close, xgb_preds_close, y_test_prev_close)
    lstm_mae, lstm_rmse, lstm_mape, lstm_dir = compute_metrics(y_test_true_close, lstm_preds_close, y_test_prev_close)
    
    metrics = {
        "val_mae": float(val_mae),
        "val_rmse": float(val_rmse),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_mape": float(test_mape),
        "test_directional_accuracy": float(dir_acc),
        "naive_baseline_mape": float(naive_mape),
        "naive_baseline_mae": float(naive_mae),
        "naive_baseline_rmse": float(naive_rmse),
        "naive_baseline_dir": 50.0, # Naive direction is practically random/flat, but actually calculate it if needed. Wait, compute_metrics calculates dir_acc. Let's use it.
        "xgb_mae": float(xgb_mae),
        "xgb_rmse": float(xgb_rmse),
        "xgb_mape": float(xgb_mape),
        "xgb_dir_acc": float(xgb_dir),
        "lstm_mae": float(lstm_mae),
        "lstm_rmse": float(lstm_rmse),
        "lstm_mape": float(lstm_mape),
        "lstm_dir_acc": float(lstm_dir)
    }
    
    # Actually naive dir_acc is defined!
    _, _, _, naive_dir = compute_metrics(y_test_true_close, y_test_prev_close, y_test_prev_close)
    metrics["naive_baseline_dir"] = float(naive_dir)
    
    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("[SUCCESS] Saved metrics/scores.json")
    return metrics

@task(name="register_if_improved")
def register_if_improved(metrics, run_id, model_name="btc-price-forecasting-ensemble"):
    print("Checking if model improved...")
    client = MlflowClient()
    new_rmse = metrics["test_rmse"]
    
    # Check for champion model
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            champion = versions[0]
            champ_run = client.get_run(champion.run_id)
            champ_rmse = champ_run.data.metrics.get("test_rmse", float("inf"))
        else:
            champ_rmse = float("inf")
    except Exception:
        champ_rmse = float("inf")
        
    print(f"New RMSE: {new_rmse:.4f} | Champion RMSE: {champ_rmse:.4f}")
    
    if new_rmse < champ_rmse * 0.98:
        print("New model beats champion by > 2%! Registering...")
        result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
        
        # Transition to staging
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging"
        )
        print(f"Model version {result.version} transitioned to Staging.")
        print("Canary: 10% traffic, 24h (Simulated)")
        print("Pass -> Transitioning to Production")
        
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )
    else:
        print("New model did not improve by > 2%. Keeping champion.")

@flow(name="btc-train")
def train():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    p_lstm = params["lstm"]
    p_xgb = params["xgboost"]
    p_ens = params["ensemble"]
    
    train_df = pd.read_csv("data/processed/splits/train.csv", index_col="Date", parse_dates=True)
    val_df = pd.read_csv("data/processed/splits/val.csv", index_col="Date", parse_dates=True)
    test_df = pd.read_csv("data/processed/splits/test.csv", index_col="Date", parse_dates=True)
    
    target = 'daily_return'
    os.makedirs("reports", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    xgb_model, xgb_val_preds, xgb_test_preds = train_xgboost(train_df, val_df, test_df, target, p_xgb)
    lstm_model, lstm_val_preds, lstm_test_preds = train_lstm(train_df, val_df, test_df, target, p_lstm)
    
    # Save the models to artifacts/ for live inference API
    joblib.dump(xgb_model, "artifacts/xgb_model.pkl")
    torch.save(lstm_model.state_dict(), "artifacts/lstm_model.pth")
    print("SUCCESS: Models saved to artifacts/")
    
    metrics = evaluate_ensemble(train_df, val_df, test_df, target, p_lstm, p_ens, lstm_val_preds, lstm_test_preds, xgb_val_preds, xgb_test_preds)
    
    # Evaluate baselines
    from src.models.baselines import evaluate_baselines
    scaler = joblib.load("artifacts/scaler.pkl")
    baseline_metrics = evaluate_baselines(train_df, val_df, test_df, target, p_lstm["sequence_length"], scaler)
    metrics.update(baseline_metrics)
    
    # Print the requested Comparison Table
    try:
        from tabulate import tabulate
        table = [
            ["Naive (Yesterday=Today)", f"${metrics['naive_baseline_mae']:,.2f}", f"{metrics['naive_baseline_mape']:.2f}%", f"{metrics['naive_baseline_dir']:.1f}%"],
            ["Linear Regression", f"${metrics['lr_mae']:,.2f}", f"{metrics['lr_mape']:.2f}%", f"{metrics['lr_dir_acc']:.1f}%"],
            ["Facebook Prophet", f"${metrics['prophet_mae']:,.2f}", f"{metrics['prophet_mape']:.2f}%", f"{metrics['prophet_dir_acc']:.1f}%"],
            ["XGBoost only", f"${metrics['xgb_mae']:,.2f}", f"{metrics['xgb_mape']:.2f}%", f"{metrics['xgb_dir_acc']:.1f}%"],
            ["LSTM only", f"${metrics['lstm_mae']:,.2f}", f"{metrics['lstm_mape']:.2f}%", f"{metrics['lstm_dir_acc']:.1f}%"],
            ["Ensemble (yours)", f"${metrics['test_mae']:,.2f}", f"{metrics['test_mape']:.2f}%", f"{metrics['test_directional_accuracy']:.1f}%"]
        ]
        print("\n" + "="*60)
        print("BASELINE COMPARISON TABLE")
        print("="*60)
        print(tabulate(table, headers=["Model", "Test MAE", "Test MAPE", "Directional Accuracy"], tablefmt="github"))
        print("="*60 + "\n")
    except Exception as e:
        print("Could not print table:", e)
    
    # DagsHub/MLflow integration
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow_tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("btc-price-forecasting")
    
    with mlflow.start_run(run_name="lstm_xgb_ensemble_v1") as run:
        mlflow.log_params({f"lstm_{k}": v for k, v in p_lstm.items()})
        mlflow.log_params({f"xgb_{k}": v for k, v in p_xgb.items()})
        mlflow.log_params({f"ens_{k}": v for k, v in p_ens.items()})
        mlflow.log_metrics(metrics)
        
        with open("metrics/scores.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metrics/scores.json")
        
        if os.path.exists("reports/shap_summary.png"):
            mlflow.log_artifact("reports/shap_summary.png")
            
        mlflow.xgboost.log_model(xgb_model, "model")
        register_if_improved(metrics, run.info.run_id)

if __name__ == "__main__":
    train()
