# who?? Cognitive Digital Twin & Behavioral Modeling Framework
FROM python:3.11-slim

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
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

# Copy source code, benchmark profiles, and entrypoint
COPY src/ ./src/
COPY profiles/ ./profiles/
COPY main.py ./

# Healthcheck to verify system integrity and diagnostic readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD who doctor || exit 1

# Default command: launch who MCP server over stdio for AI agents
ENTRYPOINT ["who"]
CMD ["mcp"]
