#!/bin/bash

# Start Flask backend server with proper SSL certificate configuration

cd "$(dirname "$0")"

# Set SSL certificate environment variables
# CRITICAL: These must be set BEFORE Python imports fsspec/gcsfs
export SSL_CERT_FILE=$(python3 -m certifi)
export REQUESTS_CA_BUNDLE=$(python3 -m certifi)
export AIOHTTP_CA_BUNDLE=$(python3 -m certifi)

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
pkill -f "python.*app.py" 2>/dev/null
sleep 2

# Start the server
echo "Starting server..."
python3 app.py

