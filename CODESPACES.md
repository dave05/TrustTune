# TrustTune on GitHub Codespaces

This document provides instructions for running TrustTune on GitHub Codespaces.

## Quick Start

1. Go to the GitHub repository: https://github.com/dave05/TrustTune
2. Click the "Code" button
3. Select the "Codespaces" tab
4. Click "Create codespace on main"
5. Wait for the codespace to initialize
6. Once ready, run the following command in the terminal:

```bash
./.devcontainer/codespace-setup.sh && uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

7. Click on the "Ports" tab and click the globe icon next to port 8000 to open the application

## Features

- Interactive web UI for calibrating ML model scores
- Multiple calibration methods (Platt, Isotonic, Temperature)
- Visualization of calibration results with reliability diagrams
- Comprehensive metrics (ECE, Brier Score)
- Sample data for testing

## Development

To run the tests:

```bash
# Run Python tests
pytest

# Run end-to-end tests
npx playwright test
```

To view the test reports:

```bash
npx playwright show-report
```

## Deployment URL

When your Codespace is running, you can access the application at:

```
https://{your-codespace-name}-8000.preview.app.github.dev/
```

Where `{your-codespace-name}` is the unique name assigned to your Codespace.

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed by running the setup script
2. Check that port 8000 is forwarded (visible in the "Ports" tab)
3. Ensure the application is running with the correct host and port
4. Check the terminal for any error messages
