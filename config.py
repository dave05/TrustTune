"""Configuration settings for TrustTune."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PythonAnywhere deployment settings
PYTHONANYWHERE_USERNAME = os.getenv("PYTHONANYWHERE_USERNAME")
PYTHONANYWHERE_DOMAIN = f"{PYTHONANYWHERE_USERNAME}.pythonanywhere.com"
PYTHONANYWHERE_API_TOKEN = os.getenv("PYTHONANYWHERE_API_TOKEN")

# Application settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key-for-development-only")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trusttune.db")

# API settings
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
