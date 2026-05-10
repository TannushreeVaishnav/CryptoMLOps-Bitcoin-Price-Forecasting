## CI/CD Pipeline

GitHub Actions workflow for continuous integration and deployment.

### Components
- `.github/workflows/ci.yml` — Linting, testing, and DVC pipeline validation on every push
- Docker Compose — Spins up MLflow, FastAPI, and Streamlit services
- `infra/` — Infrastructure configuration files

### Workflow Triggers
- Push to `master` or any `feature/*` branch
- Runs: dependency install → lint → tests → DVC check
