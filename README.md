# TrustTune: Production-Ready ML Score Calibration

TrustTune is a robust, plug-and-play solution for automating model score calibration in production ML systems. It supports both online (streaming) and offline (batch) calibration with comprehensive monitoring and drift detection.

## Features

- Multiple calibration methods:
  - Platt Scaling
  - Isotonic Regression
  - Temperature Scaling
  - Beta Calibration (coming soon)
- Online and offline calibration
- Comprehensive calibration metrics
- Drift detection and monitoring
- Model versioning and governance
- REST API and Python SDK
- Production-ready with Docker support
- Interactive web UI

## GitHub Codespaces

The easiest way to get started with TrustTune is to use GitHub Codespaces:

1. Click the "Code" button on the GitHub repository
2. Select the "Codespaces" tab
3. Click "Create codespace on main"
4. Wait for the codespace to initialize
5. Once ready, run the following command in the terminal:

```bash
./.devcontainer/codespace-setup.sh && uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

6. Click on the "Ports" tab and click the globe icon next to port 8000 to open the application

## Local Installation

```bash
# Clone the repository
git clone https://github.com/dave05/TrustTune.git
cd TrustTune

# Install dependencies
pip install -e .
pip install -r requirements.txt

# Run the application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

```bash
# Build the Docker image
docker build -t trusttune .

# Run the container
docker run -p 8000:8000 trusttune
```

## Python Usage

```python
from trusttune.core.platt import PlattCalibrator

# Initialize calibrator
calibrator = PlattCalibrator()

# Fit calibrator with your data
calibrator.fit(scores, labels)

# Get calibrated probabilities
calibrated_scores = calibrator.predict_proba(scores)
```

## API Usage

```python
import requests

# Prepare data
data = {
    "scores": [0.2, 0.7, 0.9],
    "labels": [0, 1, 1],
    "calibrator_type": "platt"
}

# Call API
response = requests.post(
    "http://localhost:8000/calibrate",
    json=data
)

# Get results
result = response.json()
print(result["calibrated_scores"])
```

## End-to-End Testing

TrustTune includes comprehensive end-to-end tests using Playwright:

```bash
# Install Playwright
npm init -y
npm install @playwright/test
npx playwright install --with-deps

# Run tests
npx playwright test
```

## Quick Start with Python

```python
from trusttune.core.platt import PlattCalibrator