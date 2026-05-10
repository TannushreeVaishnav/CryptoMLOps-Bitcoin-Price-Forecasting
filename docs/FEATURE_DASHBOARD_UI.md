## Dashboard UI

Premium Neomorphic Streamlit dashboard with 4 tabs.

### Components
- `dashboard/app.py` — Streamlit app with premium CSS injection
- `dashboard/forecast_template.html` — Area SVG chart for BTC predictions
- `dashboard/performance_template.html` — KPI cards, stability chart, run comparison table
- `dashboard/features_template.html` — SHAP feature importance dot chart + heatmap
- `dashboard/drift_template.html` — Live ingestion flow diagram and alert ledger
- `dashboard/_modal_util.html` — Shared modal component

### Run
```bash
streamlit run dashboard/app.py
```
