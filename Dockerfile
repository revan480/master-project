# Master Project: Causal Discovery for Predictive Maintenance
# Multi-stage build optimized for caching

FROM python:3.9-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Dependencies stage - cached separately
# ============================================
FROM base AS dependencies

# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies in stages for better caching
# Core dependencies first
RUN pip install --no-cache-dir \
    pandas>=1.5.0 \
    numpy>=1.23.0 \
    scipy>=1.9.0 \
    scikit-learn>=1.1.0

# Statistical and ML libraries
RUN pip install --no-cache-dir \
    statsmodels>=0.13.0 \
    networkx>=2.8.0

# Causal discovery libraries (these take longer)
RUN pip install --no-cache-dir tigramite>=5.2.0
RUN pip install --no-cache-dir lingam>=1.7.0
RUN pip install --no-cache-dir gcastle torch --extra-index-url https://download.pytorch.org/whl/cpu

# Visualization and utilities
RUN pip install --no-cache-dir \
    matplotlib>=3.6.0 \
    seaborn>=0.12.0 \
    tqdm>=4.64.0 \
    pyyaml>=6.0 \
    click>=8.1.0


# ============================================
# Final stage
# ============================================
FROM dependencies AS final

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/results \
    && chown -R appuser:appuser /app

# Copy source code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Default command
CMD ["python", "-m", "src.algorithms.algorithm_runner", "--help"]
