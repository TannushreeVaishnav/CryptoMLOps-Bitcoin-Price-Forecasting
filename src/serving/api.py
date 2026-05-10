from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Bitcoin Price Forecasting API",
    description="API for serving Bitcoin price predictions using an ensemble of LSTM and XGBoost.",
    version="1.0.0"
)

class FeaturesInput(BaseModel):
    # Using a generic list for the 30-day sequence features for now
    data: list

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Bitcoin Price Forecasting API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

from fastapi.responses import FileResponse
import json
import os

@app.get("/metrics")
def get_metrics():
    if os.path.exists("metrics/scores.json"):
        with open("metrics/scores.json", "r") as f:
            return json.load(f)
    return {"error": "Metrics file not found. Train the model first."}

@app.get("/shap")
def get_shap():
    if os.path.exists("reports/shap_summary.png"):
        return FileResponse("reports/shap_summary.png", media_type="image/png")
    return {"error": "SHAP summary plot not found."}

@app.post("/predict")
def predict(input_data: FeaturesInput):
    try:
        from src.data.loader import fetch_live_btc
        from src.data.features import engineer_features
        import joblib
        import torch
        import pandas as pd
        import yaml
        from src.models.lstm_model import BTCLSTMModel
        from src.models.ensemble import ensemble_predict
        import os
        
        # Load params for weights and sequence length
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
            
        seq_len = params["lstm"]["sequence_length"]
        lstm_w = params["ensemble"]["lstm_weight"]
        xgb_w = params["ensemble"]["xgb_weight"]
        
        # 1. Fetch Live Data (100 days to survive dropna and have 30 days left)
        df = fetch_live_btc(days=100)
        latest_close = float(df['Close'].iloc[-1])
        
        # 2. Engineer Features
        df_features = engineer_features(df)
        
        # 3. Scale Features
        scaler = joblib.load("artifacts/scaler.pkl")
        scaled_features = scaler.transform(df_features)
        df_scaled = pd.DataFrame(scaled_features, index=df_features.index, columns=df_features.columns)
        
        # 4. Extract sequences
        x_xgb = df_scaled.iloc[-1:].copy()
        
        x_lstm_df = df_scaled.iloc[-seq_len:]
        x_lstm = torch.tensor(x_lstm_df.values, dtype=torch.float32).unsqueeze(0) # (1, 30, num_features)
        
        # 5. Load Models
        xgb_model = joblib.load("artifacts/xgb_model.pkl")
        
        num_features = x_lstm_df.shape[1]
        lstm_model = BTCLSTMModel(
            input_size=num_features,
            hidden_size=params["lstm"]["hidden_size"],
            num_layers=params["lstm"]["num_layers"],
            dropout=params["lstm"]["dropout"]
        )
        lstm_model.load_state_dict(torch.load("artifacts/lstm_model.pth"))
        lstm_model.eval()
        
        # 6. Predict
        xgb_pred = xgb_model.predict(x_xgb)
        with torch.no_grad():
            lstm_pred = lstm_model(x_lstm).numpy()
            
        pred_return_scaled = ensemble_predict(lstm_pred, xgb_pred, lstm_w, xgb_w)[0]
        
        # 7. Unscale
        target_idx = list(df_features.columns).index("daily_return")
        target_mean = scaler.mean_[target_idx]
        target_scale = scaler.scale_[target_idx]
        
        pred_return = pred_return_scaled * target_scale + target_mean
        predicted_close = latest_close * (1 + pred_return)
        
        prediction = {
            "latest_close": round(latest_close, 2),
            "predicted_close": round(predicted_close, 2),
            "predicted_return": round(float(pred_return) * 100, 2),
            "directional_movement": "UP" if pred_return > 0 else "DOWN"
        }
        return {"status": "success", "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)
