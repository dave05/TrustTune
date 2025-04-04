
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

ENV PORT=8000
EXPOSE ${PORT}

# Set up entrypoint for development
CMD ["uvicorn", "trusttune.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

