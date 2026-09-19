"""Logging configuration."""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

from trusttune.config import settings

def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = RotatingFileHandler(
        log_dir / "trusttune.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # Sentry integration
    if settings.ENVIRONMENT == "production":
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            integrations=[
                LoggingIntegration(
                    level=logging.INFO,
                    event_level=logging.ERROR
                )
            ],
            environment=settings.ENVIRONMENT,
            traces_sample_rate=1.0
        )

    return root_logger