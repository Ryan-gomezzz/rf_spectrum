FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir openenv-core pydantic uvicorn fastapi

# Copy environment code
COPY . .

# Expose port
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
