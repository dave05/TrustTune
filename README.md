# TrustTune: Production-Ready ML Score Calibration

TrustTune is a robust, plug-and-play solution for automating model score calibration in production ML systems. It supports both online (streaming) and offline (batch) calibration with comprehensive monitoring and drift detection.

## Features

- Multiple calibration methods:
  - Platt Scaling
  - Isotonic Regression (coming soon)
  - Temperature Scaling (coming soon)
  - Beta Calibration (coming soon)
- Online and offline calibration
- Comprehensive calibration metrics
- Drift detection and monitoring
- Model versioning and governance
- REST API and Python SDK
- Production-ready with Docker support

## Quick Start

```python
from trusttune.core.platt import PlattCalibrator 