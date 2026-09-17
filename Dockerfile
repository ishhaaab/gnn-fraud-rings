FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACT_DIR=/app/results/serving

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    && useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser api/ api/
COPY --chown=appuser:appuser results/serving/ results/serving/

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
