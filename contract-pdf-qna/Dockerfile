# Dockerfile for contract-pdf-qna application
# Uses Python 3.11.7 as specified in .python-version

FROM python:3.11.14

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    # Force CPU-only PyTorch (prevents installing NVIDIA CUDA runtime packages)
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch && \
    # Install the rest (prefer wheels to avoid building from source)
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p files

# Expose port (app runs on port 8001)
EXPOSE 8001

# Set SSL certificate environment variables
# These are critical for fsspec/gcsfs to work properly
ENV SSL_CERT_FILE=/usr/local/lib/python3.11/site-packages/certifi/cacert.pem \
    REQUESTS_CA_BUNDLE=/usr/local/lib/python3.11/site-packages/certifi/cacert.pem \
    AIOHTTP_CA_BUNDLE=/usr/local/lib/python3.11/site-packages/certifi/cacert.pem

# Health check (adjust based on your app's health endpoint if available)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["python", "app.py"]

