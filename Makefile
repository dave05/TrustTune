.PHONY: test format lint check clean

test:
    pytest -v --cov=trusttune --cov-report=term-missing tests/

format:
    black .
    isort .

lint:
    mypy trusttune

check:
    ./scripts/run_checks.sh

clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    find . -type f -name ".coverage" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type d -name "*.egg" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
    find . -type d -name "build" -exec rm -rf {} +
    find . -type d -name "dist" -exec rm -rf {} +