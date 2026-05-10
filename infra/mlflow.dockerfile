FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow_tracking.db"]
