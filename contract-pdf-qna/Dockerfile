# Backend Dockerfile (Flask)
# Builds an image that runs contract-pdf-qna/app.py on port 8001.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps required by some Python packages (cryptography, lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8001

# NOTE: Auth/GCP/OpenAI keys must be provided as environment variables at runtime.
CMD ["python", "app.py"]


