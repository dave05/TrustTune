#!/bin/bash
set -e

echo "Setting up TrustTune in GitHub Codespaces..."

# Install dependencies
pip install -e .
pip install -r requirements.txt
pip install -r trusttune/requirements-dev.txt

# Install Playwright dependencies
npm init -y
npm install @playwright/test
npx playwright install --with-deps

# Create necessary directories
mkdir -p static/js static/css templates

# Copy static files if they don't exist in the root
if [ ! -f "static/js/main.js" ]; then
    cp -r /app/static/* static/
fi

if [ ! -f "templates/index.html" ]; then
    cp -r /app/templates/* templates/
fi

# Set up environment variables
export PORT=8000

echo "Setup complete! Run 'uvicorn app:app --host 0.0.0.0 --port 8000 --reload' to start the application."
