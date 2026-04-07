FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Copy lockfile and manifest first for layer caching
COPY pyproject.toml uv.lock ./

# Install all dependencies from the frozen lockfile (no network guessing)
RUN uv sync --no-dev --frozen

# Copy the rest of the project
COPY . .

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
