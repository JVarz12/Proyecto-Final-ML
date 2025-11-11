FROM python:3.13-slim

WORKDIR /app

# Copiamos archivos de configuración y requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código fuente
COPY mlops_pipeline ./mlops_pipeline
COPY app ./app

ENV PYTHONPATH="/app/mlops_pipeline/src"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
