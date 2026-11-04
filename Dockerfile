# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.9.26 AS uv

FROM python:3.12.16-slim-trixie AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build
COPY deployment/api-requirements.lock ./api-requirements.lock
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --require-hashes --no-deps \
       --no-cache -r api-requirements.lock

FROM python:3.12.16-slim-trixie AS runtime
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUD_HOST=0.0.0.0 \
    FRAUD_PORT=8000

RUN groupadd --system --gid 10001 fraud \
    && useradd --system --uid 10001 --gid fraud --home-dir /nonexistent --shell /usr/sbin/nologin fraud

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=fraud:fraud src ./src
COPY --chown=fraud:fraud alembic ./alembic
COPY --chown=fraud:fraud alembic.ini ./alembic.ini

USER 10001:10001
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).read()"]

CMD ["uvicorn", "fraud_detection.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "30", "--no-access-log"]
