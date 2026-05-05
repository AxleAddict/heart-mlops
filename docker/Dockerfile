# ---- build stage ----
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime stage ----
FROM python:3.11-slim
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code and model
COPY src/ src/
COPY models/ models/

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV HEART_MODEL_PATH=/app/models/model.pkl
ENV LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "heart.api:app", "--host", "0.0.0.0", "--port", "8000"]
