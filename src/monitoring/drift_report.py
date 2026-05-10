import os
# Fix OpenBLAS Memory Error on Windows/Low RAM systems
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import json
import os
from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset

def run_drift_report():
    print("Generating Evidently Drift Report...")
    # Load data
    train_df = pd.read_csv("data/processed/splits/train.csv", index_col="Date", parse_dates=True)
    test_df = pd.read_csv("data/processed/splits/test.csv", index_col="Date", parse_dates=True)
    
    # Just an example of getting the feature columns
    target = 'daily_return'
    features = [c for c in train_df.columns if c != target]
    
    reference_df = train_df[features].copy()
    current_df = test_df[features].copy()
    
    # Initialize Evidently Report
    report = Report(metrics=[
        DataDriftPreset(),
        # RegressionPreset() can be added if we include predictions
    ])
    
    snapshot = report.run(reference_data=reference_df, current_data=current_df)
    
    os.makedirs("reports", exist_ok=True)
    
    # Save HTML format for human viewing
    snapshot.save_html("reports/drift.html")
    
    # Save JSON format for dashboard parsing
    snapshot.save_json("reports/drift.json")
    
    print("[SUCCESS] Drift report saved to reports/drift.json and reports/drift.html")
    
    # Evaluate Monitoring Thresholds
    try:
        with open("reports/drift.json", "r") as f:
            drift_data = json.load(f)
            
        metrics = drift_data.get("metrics", [])
        for m in metrics:
            config = m.get("config", {})
            if config.get("type") == "evidently:metric_v2:ValueDrift":
                col = config.get("column")
                psi = m.get("value", 0)
                if col and psi > 0.2:
                    print(f"ALERT: Feature drift (PSI > 0.2) detected on {col}: {psi:.4f}")
                    # In production: Trigger unscheduled retraining via Prefect or Slack alert here!
    except Exception as e:
        print(f"Error parsing drift JSON for alerts: {e}")

if __name__ == "__main__":
    run_drift_report()
