## Drift Monitoring

Automated data drift and concept drift detection using Evidently AI.

### Components
- `src/monitoring/drift_report.py` — Generates Evidently drift report
- `dashboard/drift_template.html` — Live PSI gauge, distribution shift charts, alert ledger
- `reports/` — JSON and HTML drift reports (gitignored, regenerated on demand)

### Run
```bash
python src/monitoring/drift_report.py
```
