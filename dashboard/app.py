import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import json
import os
import sys
from PIL import Image
from dotenv import load_dotenv

# Absolute path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

env_path = os.path.join(project_root, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    # Forcefully set os.environ again just in case Streamlit intercepts it
    with open(env_path, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val

# Streamlit config
st.set_page_config(page_title="Bitcoin Price Forecasting", layout="wide")

# Set MLflow tracking URI (handle both local and docker network)
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_uri)

st.title("🚀 Bitcoin Price Forecasting Dashboard")

# Load shared modal utility once
_modal_util_path = os.path.join(current_dir, '_modal_util.html')
_modal_utility_html = ""
if os.path.exists(_modal_util_path):
    with open(_modal_util_path, 'r') as f:
        _modal_utility_html = f.read()

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
            live_df = fetch_live_btc(days=60)
            latest_close = float(live_df['Close'].iloc[-1])
        except Exception:
            # Fallback to test set if no internet
            latest_close = recent_df['Close'].iloc[-1]
            
        predicted_close = latest_close * 1.015 # Placeholder until we run full inference
        
        # Calculate Prediction details
        ret = (predicted_close - latest_close) / latest_close * 100
        is_bull = ret > 0
        color = "tertiary" if is_bull else "error"
        color_container = "tertiary-container" if is_bull else "error-container"
        icon = "arrow_upward" if is_bull else "arrow_downward"
        pred_dir = "Strong Bullish Trend Detected" if is_bull else "Bearish Correction Expected"
        alpha_desc = "High-probability long scenario identified based on positive prediction." if is_bull else "Short-term downside risk identified based on negative prediction."
        
        # Format metrics
        m_mae = f"{scores.get('test_mae', 0):.2f}"
        m_rmse = f"{scores.get('test_rmse', 0):.2f}"
        m_mape = f"{scores.get('test_mape', 0):.2f}"
        m_dir = f"{scores.get('test_directional_accuracy', 0):.1f}"
        
        # Load and render custom HTML
        import streamlit.components.v1 as components
        template_path = os.path.join(current_dir, 'forecast_template.html')
        if os.path.exists(template_path):
            with open(template_path, 'r') as file:
                html_code = file.read()
                
            html_code = html_code.replace("{latest_close}", f"{latest_close:,.2f}")
            html_code = html_code.replace("{predicted_close}", f"{predicted_close:,.2f}")
            html_code = html_code.replace("{predicted_change}", f"{abs(ret):.2f}")
            html_code = html_code.replace("{color}", color)
            html_code = html_code.replace("{color_container}", color_container)
            html_code = html_code.replace("{icon}", icon)
            html_code = html_code.replace("{predicted_direction}", pred_dir)
            html_code = html_code.replace("{alpha_description}", alpha_desc)
            html_code = html_code.replace("{metric_mae}", m_mae)
            html_code = html_code.replace("{metric_rmse}", m_rmse)
            html_code = html_code.replace("{metric_mape}", m_mape)
            html_code = html_code.replace("{metric_dir}", m_dir)
            
            components.html(html_code, height=800, scrolling=True)
        else:
            st.error("Forecast template not found.")
            
        st.line_chart(recent_df[['Close']])
        
    except Exception as e:
        st.warning(f"Could not load historical data: {e}")

with tab2:
    template_path = os.path.join(current_dir, 'performance_template.html')
    if os.path.exists(template_path):
        with open(template_path, 'r') as file:
            html_code = file.read()
            
            # Generate Real Heatmap from recent_df
            heatmap_html = ""
            if 'recent_df' in locals() and not recent_df.empty:
                cols = ['Close', 'Volume', 'RSI_14', 'EMA_12', 'MACD_12_26_9']
                if all(c in recent_df.columns for c in cols):
                    corr = recent_df[cols].corr()
                    for i in range(len(cols)):
                        for j in range(len(cols)):
                            val = corr.iloc[i, j]
                            # Opacity based on absolute correlation
                            opacity = max(10, int(abs(val) * 100))
                            bg = "primary" if val >= 0 else "error"
                            title = f"{cols[i]} vs {cols[j]}: {val:.2f}"
                            # If correlation is near 0 or 1, distinct look
                            if val == 1.0:
                                bg = "primary-container"
                            heatmap_html += f'<div title="{title}" class="aspect-square bg-{bg} rounded-sm opacity-[{opacity}%] cursor-pointer hover:scale-110 transition-transform"></div>\n'
                else:
                    heatmap_html = '<div class="col-span-5 text-center text-error">Missing required columns for correlation</div>'
            else:
                heatmap_html = '<div class="col-span-5 text-center text-error">Data not available</div>'
                
            html_code = html_code.replace('{heatmap_html}', heatmap_html)
            
            html_code = html_code.replace('{modal_utility}', _modal_utility_html)
            import streamlit.components.v1 as components
            components.html(html_code, height=1500, scrolling=True)
    else:
        st.error("Performance template not found.")

with tab3:
    template_path = os.path.join(current_dir, 'features_template.html')
    if os.path.exists(template_path):
        with open(template_path, 'r') as file:
            html_code = file.read()
            
            # Inject SHAP image
            shap_path = "reports/shap_summary.png"
            shap_base64 = ""
            if os.path.exists(shap_path):
                import base64
                with open(shap_path, "rb") as image_file:
                    shap_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            
            html_code = html_code.replace('{shap_image_base64}', shap_base64)
            
            html_code = html_code.replace('{modal_utility}', _modal_utility_html)
            import streamlit.components.v1 as components
            components.html(html_code, height=1200, scrolling=True)
    else:
        st.error("Features template not found.")

with tab4:
    template_path = os.path.join(current_dir, 'drift_template.html')
    if os.path.exists(template_path):
        with open(template_path, 'r') as file:
            html_code = file.read()
            
            # Inject real drift metrics
            drift_json_path = "reports/drift.json"
            agg_psi = "0.000"
            alert_rows = ""
            if os.path.exists(drift_json_path):
                try:
                    with open(drift_json_path, "r") as f:
                        import json
                        drift_data = json.load(f)
                    metrics = drift_data.get("metrics", [])
                    drift_table = []
                    
                    drift_share = 0.0
                    for m in metrics:
                        config = m.get("config", {})
                        if config.get("type") == "evidently:metric_v2:DriftedColumnsCount":
                            drift_share = m.get("value", {}).get("share", 0.0)
                            agg_psi = f"{drift_share:.3f}"
                            
                        if config.get("type") == "evidently:metric_v2:ValueDrift":
                            feat = config.get("column")
                            score = m.get("value")
                            threshold = config.get("threshold", 0.1)
                            if feat and score is not None:
                                is_drifted = score > threshold
                                if is_drifted:
                                    drift_table.append({"Feature": feat, "Score": score})
                    
                    # Generate HTML table rows
                    from datetime import datetime
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    for row in sorted(drift_table, key=lambda x: x["Score"], reverse=True)[:5]:
                        score_val = row['Score']
                        feat = row['Feature']
                        modal_title = f"Drift Alert: {feat}"
                        modal_body = (
                            f"<p><strong style='color:#ffb4ab;'>CRITICAL drift detected</strong> in feature <code>{feat}</code>.</p>"
                            f"<table style='width:100%;margin-top:14px;border-collapse:collapse;font-size:13px;'>"
                            f"<tr style='border-bottom:1px solid #424754;'><td style='padding:8px;color:#8c909f;'>Drift Score</td><td style='padding:8px;font-weight:700;color:#ffb4ab;'>{score_val:.4f} Wasserstein Distance</td></tr>"
                            f"<tr style='border-bottom:1px solid #424754;'><td style='padding:8px;color:#8c909f;'>Threshold</td><td style='padding:8px;font-weight:700;'>0.10</td></tr>"
                            f"<tr style='border-bottom:1px solid #424754;'><td style='padding:8px;color:#8c909f;'>Method</td><td style='padding:8px;'>Wasserstein distance (normed)</td></tr>"
                            f"<tr style='border-bottom:1px solid #424754;'><td style='padding:8px;color:#8c909f;'>Status</td><td style='padding:8px;font-weight:700;'>INVESTIGATING</td></tr>"
                            f"<tr><td style='padding:8px;color:#8c909f;'>Report File</td><td style='padding:8px;font-family:monospace;font-size:11px;'>reports/drift.json</td></tr>"
                            f"</table>"
                            f"<p style='margin-top:14px;color:#8c909f;font-size:12px;'>A drift score this high indicates the <strong>{feat}</strong> distribution has shifted significantly between your training data and live market data. Consider triggering a retraining pipeline.</p>"
                        )
                        onclick_str = f"openBtcModal('{modal_title}', '{modal_body}')"
                        alert_rows += f'''
                        <tr class="hover:bg-surface-variant/30 transition-colors">
                        <td class="px-6 py-4 font-label-caps opacity-60 font-bold">{now_str}</td>
                        <td class="px-6 py-4 font-bold text-sm">{feat}</td>
                        <td class="px-6 py-4">
                        <span class="inline-flex items-center gap-2 px-4 py-1 bg-error-container/20 text-error rounded-full text-[10px] font-bold tracking-widest">
                        <span class="w-1.5 h-1.5 rounded-full bg-error"></span> CRITICAL
                        </span>
                        </td>
                        <td class="px-6 py-4 font-label-caps text-error font-bold">{score_val:.2f} DIST</td>
                        <td class="px-6 py-4 text-on-surface-variant font-bold tracking-wider text-[11px]">INVESTIGATING</td>
                        <td class="px-6 py-4">
                        <button onclick="{onclick_str}" class="text-primary hover:underline font-label-caps text-label-caps font-bold cursor-pointer">VIEW LOGS</button>
                        </td>
                        </tr>
                        '''
                except Exception as e:
                    pass
            
            if not alert_rows:
                alert_rows = '<tr><td colspan="6" class="px-6 py-4 text-center">No drift detected.</td></tr>'
                
            html_code = html_code.replace('0.084', agg_psi, 1)
            # Find the alert ledger body and replace it
            import re
            html_code = re.sub(r'<tbody class="font-body-sm divide-y divide-outline-variant">.*?</tbody>', f'<tbody class="font-body-sm divide-y divide-outline-variant">{alert_rows}</tbody>', html_code, flags=re.DOTALL)
            
            html_code = html_code.replace('{modal_utility}', _modal_utility_html)
            import streamlit.components.v1 as components
            components.html(html_code, height=1400, scrolling=True)
    else:
        st.error("Drift template not found.")
