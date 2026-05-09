FROM python:3.11-slim

WORKDIR /app

COPY webapp/requirements.txt /app/webapp/requirements.txt
RUN pip install --no-cache-dir -r /app/webapp/requirements.txt

COPY ml /app/ml
COPY webapp /app/webapp

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/results/models/design_rf_model.joblib

# Optional: copy model at build if present in build context
RUN mkdir -p /app/results/models

EXPOSE 8000

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
