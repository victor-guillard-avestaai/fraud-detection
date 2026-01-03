# Dockerfile (prod image for Cloud Run)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

# 1) Dependency layer
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# 2) Source code
COPY internalpy/ /app/internalpy/
COPY api/        /app/api/
# Model artifacts baked into the image (small files only)
COPY model/artifacts/ /app/model/artifacts/
COPY model/demo_samples.json /app/model/demo_samples.json
# Static demo UI
COPY static/ /app/static/
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8080

# Default command: run FastAPI app with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
