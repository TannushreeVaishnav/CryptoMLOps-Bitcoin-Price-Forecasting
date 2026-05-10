import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import json
import os
from PIL import Image

# Streamlit config
st.set_page_config(page_title="Bitcoin Price Forecasting", layout="wide")

# Set MLflow tracking URI (handle both local and docker network)
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_uri)

st.title("🚀 Bitcoin Price Forecasting Dashboard")

# Create tabs for pages
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 Model Performance", "🧠 Feature Importance", "🚨 Drift Status"])

with tab1:
    st.header("BTC Actual vs Predicted (Last 90 Days)")
    
    # Load actual data
    try:
        df = pd.read_csv("data/processed/splits/test.csv", index_col="Date", parse_dates=True)
        recent_df = df.tail(90)
        
        # Load latest metrics
        with open("metrics/scores.json", "r") as f:
            scores = json.load(f)
            
        st.subheader("Tomorrow's Prediction")
        
        import joblib
        # Load scaler to inverse transform the Close price for the Line Chart
        if os.path.exists("artifacts/scaler.pkl"):
            scaler = joblib.load("artifacts/scaler.pkl")
            # Inverse transform Close
            close_idx = list(df.columns).index('Close')
            close_mean = scaler.mean_[close_idx]
            close_scale = scaler.scale_[close_idx]
            recent_df['Close'] = recent_df['Close'] * close_scale + close_mean
        
        from src.data.loader import fetch_live_btc
        
        # Load the TRUE live internet data for the display cards
        try:
            live_df = fetch_live_btc(days=5)
            latest_close = float(live_df['Close'].iloc[-1])
        except Exception:
            # Fallback to test set if no internet
            latest_close = recent_df['Close'].iloc[-1]
            
        predicted_close = latest_close * 1.015 # Just a placeholder
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Live Actual Close", f"${latest_close:,.2f}")
        col2.metric("Predicted Close (Tomorrow)", f"${predicted_close:,.2f}", "+1.5%")
        
        direction = "Up ⬆️" if predicted_close > latest_close else "Down ⬇️"
        col3.metric("Predicted Direction", direction)
        
        st.line_chart(recent_df[['Close']])
        
    except Exception as e:
        st.warning(f"Could not load historical data: {e}")

with tab2:
    st.header("Model Performance Over Time")
    try:
        experiment = mlflow.get_experiment_by_name("btc-price-forecasting")
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            if not runs.empty and 'metrics.test_rmse' in runs.columns:
                # Sort by start time to show progression
                runs = runs.sort_values("start_time")
                
                chart_data = runs[['metrics.test_mae', 'metrics.test_rmse', 'metrics.test_mape']].dropna()
                chart_data = chart_data.rename(columns={
                    'metrics.test_mae': 'MAE',
                    'metrics.test_rmse': 'RMSE',
                    'metrics.test_mape': 'MAPE'
                })
                chart_data = chart_data.reset_index(drop=True)
                
                st.line_chart(chart_data)
            else:
                st.info("No runs with metrics found in MLflow yet.")
        else:
            st.info("Experiment not found in MLflow.")
    except Exception as e:
        st.warning(f"Could not connect to MLflow at {mlflow_uri}: {e}")

with tab3:
    st.header("Feature Importance (SHAP)")
    st.markdown("Top 10 most influential features for the latest model.")
    
    shap_path = "reports/shap_summary.png"
    if os.path.exists(shap_path):
        image = Image.open(shap_path)
        st.image(image, caption='SHAP Summary Plot', use_container_width=True)
    else:
        st.info("SHAP summary plot not found. Run training pipeline to generate it.")

with tab4:
    st.header("Data Drift Status (Evidently)")
    st.markdown("Monitoring data distribution changes (PSI) between training and live data.")
    
    drift_json_path = "reports/drift.json"
    if os.path.exists(drift_json_path):
        try:
            with open(drift_json_path, "r") as f:
                drift_data = json.load(f)
            
            # This extracts drift metrics if evidently JSON is properly saved
            metrics = drift_data.get("metrics", [])
            drift_table = []
            
            for m in metrics:
                if m.get("metric") == "DataDriftTable":
                    features = m.get("result", {}).get("drift_by_columns", {})
                    for feat, details in features.items():
                        drift_table.append({
                            "Feature": feat,
                            "Drift Score (PSI)": details.get("drift_score"),
                            "Drift Detected": "🚨 Yes" if details.get("drift_detected") else "✅ No"
                        })
            
            if drift_table:
                st.dataframe(pd.DataFrame(drift_table).sort_values("Drift Score (PSI)", ascending=False))
            else:
                st.info("Drift JSON found but no DataDriftTable metrics present.")
                
        except Exception as e:
            st.warning(f"Could not parse drift report: {e}")
    else:
        st.info("No drift report found. Run `src/monitoring/drift_report.py` to generate.")
        
    st.subheader("Monitoring Thresholds")
    st.markdown("""
    * **Feature drift (PSI) > 0.2** on top-10 features -> *Trigger unscheduled retraining*
    * **Live 7-day MAPE > 5%** -> *Retrain + Slack alert*
    * **Model age > 30 days** -> *Schedule immediate retrain*
    * **API error rate > 1%** -> *Investigate predictor logs*
    """)
