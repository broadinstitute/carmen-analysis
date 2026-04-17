# syntax=docker/dockerfile:1.7
# Multi-stage build for the CARMEN analysis web app.
# Final image runs the Streamlit UI; the same image also contains the
# `carmen-analyze` CLI entry point.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY carmen_analysis ./carmen_analysis

RUN pip install --upgrade pip wheel \
    && pip wheel --wheel-dir /wheels ".[web]"

# ------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Pull security updates that haven't yet rolled into the upstream slim
# image (Trivy gates the build on fixable HIGH/CRITICAL CVEs).
RUN apt-get update \
    && apt-get -y upgrade \
    && rm -rf /var/lib/apt/lists/*

# Streamlit needs no system libs beyond what slim provides; pandas/numpy
# wheels ship with everything needed at runtime.
RUN useradd --create-home --uid 1000 carmen

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels carmen-analysis[web] \
    && rm -rf /wheels

USER carmen

EXPOSE 8080

# Cloud Run honours $PORT; default to 8080 to match the env above.
CMD ["sh", "-c", "streamlit run --server.port=${PORT:-8080} --server.address=0.0.0.0 $(python -c 'import carmen_analysis.web.app as a; print(a.__file__)')"]
