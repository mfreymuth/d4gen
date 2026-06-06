FROM python:3.14-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . /app
ENV PATH="/app/.venv/bin:$PATH" \
	PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1
EXPOSE 80
CMD ["uv", "run", "streamlit", "run", "src/app.py", \
	"--server.port=80", \
	"--server.address=0.0.0.0", \
	"--server.headless=true"]
