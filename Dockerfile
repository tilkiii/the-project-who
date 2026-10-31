# who?? Cognitive Digital Twin & Behavioral Modeling Framework
FROM python:3.11-slim

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (build-essential, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for optimal Docker caching
COPY pyproject.toml README.md ./

# Install project and dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -e .

# Copy source code and default assets
COPY src/ ./src/
COPY profiles/ ./profiles/

# Expose Web Dashboard Port
EXPOSE 8765

# Healthcheck to verify dashboard availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8765/api/status || exit 1

# Default command: launch who UI dashboard accessible from outside the container
ENTRYPOINT ["who"]
CMD ["ui", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
