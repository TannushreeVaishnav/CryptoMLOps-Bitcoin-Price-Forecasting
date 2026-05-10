FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# Install all requirements using the extra-index-url for torch to avoid massive CUDA downloads
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
