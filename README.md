# ₿ Bitcoin Price Forecasting Platform

<div align="center">

![Premium UI Dashboard](https://img.shields.io/badge/UI-Premium_Neomorphism-black?style=for-the-badge&logo=appveyor)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)

An end-to-end Machine Learning Operations (MLOps) platform predicting Bitcoin prices utilizing an advanced **LSTM + XGBoost Ensemble Model**. Built with a state-of-the-art UI dashboard, fully reproducible data pipelines via DVC, experiment tracking with MLflow, and continuous integration via GitHub Actions.

</div>

---

## ✨ Features

- **🧠 Ensemble Modeling:** Synergistic prediction blending Deep Learning (LSTMs) for sequence extraction and Gradient Boosting (XGBoost) for tabular features.
- **🎨 Premium Neomorphic Dashboard:** A stunning, full-screen Streamlit web application providing realtime forecasts, model drift detection, feature importance attribution (SHAP), and model performance metrics.
- **🔄 Automated MLOps Pipeline:** Seamless end-to-end automation via DVC workflows, triggering data processing, training, and evaluation seamlessly.
- **📊 Experiment Tracking:** Detailed hyperparameter tuning, metric tracking, and model registry via MLflow.
- **🛡️ Drift Detection:** Live feature distribution and concept drift monitoring utilizing Evidently AI.
- **🚀 API Serving:** Robust model inference endpoints deployed via FastAPI and Uvicorn.
- **🐳 Containerized:** Ready for deployment anywhere via Docker Compose.

---

## 🛠️ Tech Stack

### Machine Learning & Data
`TensorFlow/Keras`, `XGBoost`, `Pandas`, `NumPy`, `Scikit-Learn`, `SHAP`, `Evidently`

### Backend & Infrastructure
`FastAPI`, `Uvicorn`, `Prefect`, `Docker`, `Docker Compose`

### Frontend UI
`Streamlit`, `Tailwind CSS`, `HTML5`, `Vanilla JS`, `Neomorphic Design Principles`

### MLOps
`DVC` (Data Version Control), `MLflow` (Experiment Tracking), `GitHub Actions` (CI/CD)

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/vaishnavtannushree/CryptoMLOps-Bitcoin-Price-Forecasting.git
cd CryptoMLOps-Bitcoin-Price-Forecasting

# Create Virtual Environment
python -m venv venv
venv\Scripts\activate # On Windows
# source venv/bin/activate # On Unix

# Install Dependencies
pip install -r requirements.txt
```

### 2. DVC Pipeline
Reproduce the entire machine learning pipeline (Data Ingestion ➡️ Preprocessing ➡️ Training ➡️ Evaluation):
```bash
dvc repro
```

### 3. Start the Ecosystem
Spin up the Streamlit Dashboard, FastAPI inference server, and MLflow tracking server via Docker Compose:
```bash
docker-compose up -d --build
```

### 4. Access the Applications
- **Dashboard:** `http://localhost:8501`
- **FastAPI Swagger Docs:** `http://localhost:8000/docs`
- **MLflow UI:** `http://localhost:5000`

---

## 📈 Dashboard Previews

Our Streamlit dashboard uses injected Tailwind CSS and custom HTML components for a rich application experience:
- **Forecast Tab:** Real-time Bitcoin price charting using area SVGs with dynamic tooltips.
- **Performance Tab:** MAE, Directional Accuracy, and Stability Score KPIs rendered in modern bento-box layouts.
- **Features Tab:** Real-time SHAP analysis explaining prediction local impacts.
- **Drift Tab:** Live telemetry data identifying covariate and target drift via Data Integrity alerts.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
