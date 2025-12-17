#!/bin/bash

# Start Flask backend server with proper SSL certificate configuration

cd "$(dirname "$0")"

# Prefer the local venv if present (this repo vendors `venv/`).
if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

# Choose python from the (possibly activated) environment.
# On many macOS setups, `python` is not available but `python3` is.
if [ -n "${PYTHON_BIN:-}" ]; then
  : # honor PYTHON_BIN if explicitly set
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "ERROR: Neither 'python' nor 'python3' found on PATH. Please install Python 3 and retry."
  exit 1
fi

# Set SSL certificate environment variables
# CRITICAL: These must be set BEFORE Python imports fsspec/gcsfs
CERT_PATH="$($PYTHON_BIN -m certifi 2>/dev/null)"
if [ -n "$CERT_PATH" ]; then
  export SSL_CERT_FILE="$CERT_PATH"
  export REQUESTS_CA_BUNDLE="$CERT_PATH"
  export AIOHTTP_CA_BUNDLE="$CERT_PATH"
fi

echo "=========================================="
echo "Starting Flask backend server..."
echo "=========================================="
echo "SSL Certificates configured:"
echo "  SSL_CERT_FILE: $SSL_CERT_FILE"
echo "  REQUESTS_CA_BUNDLE: $REQUESTS_CA_BUNDLE"
echo "  AIOHTTP_CA_BUNDLE: $AIOHTTP_CA_BUNDLE"
echo ""

# Kill any existing server processes
echo "Stopping any existing server processes..."
# Try to stop any previous instance of this backend (covers python/python3/Python.app).
pkill -f "[Pp]ython.*app\\.py" 2>/dev/null || true
sleep 2

# Start the server
echo "Starting server..."
$PYTHON_BIN app.py

