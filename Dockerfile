FROM python:3.14-slim AS base

# libgomp1 is required by xgboost at runtime (provides libgomp.so.1, the
# OpenMP runtime used by libxgboost.so). Without it the container will fail
# to import xgboost.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application sources + Benjamin's pre-trained XGBoost models.
# The Streamlit bootstrap registers `pkl/*.pkl` into
# `data/models/shared/benjamin_v1/` on first run inside the container.
COPY src ./src
COPY pkl ./pkl
COPY data/seed ./data/seed

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 80
CMD ["uv", "run", "streamlit", "run", "src/app.py", \
     "--server.port=80", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
