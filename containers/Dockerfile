FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    MPLBACKEND=Agg

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra workflow --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra workflow

WORKDIR /work
CMD ["cardioomics", "--help"]
