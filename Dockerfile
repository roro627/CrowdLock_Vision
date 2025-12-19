# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12-slim-bookworm

FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ULTRALYTICS_AUTOUPDATE=0

WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install --prefix=/install -r requirements.txt

FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ULTRALYTICS_AUTOUPDATE=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /install /usr/local
COPY backend /app/backend
COPY config/backend.config.example.yml /app/config/backend.config.yml

RUN mkdir -p /tmp/matplotlib /tmp/ultralytics \
    && chown -R nobody:nogroup /app /tmp/matplotlib /tmp/ultralytics \
    && chmod -R 777 /tmp/matplotlib /tmp/ultralytics

ENV PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics

EXPOSE 8000
USER nobody
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
