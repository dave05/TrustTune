import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# Get username from environment variable or use default
username = os.getenv('PYTHONANYWHERE_USERNAME', 'dawitBeshah')

# Add the project directory to the Python path
path = f'/home/{username}/TrustTune'
if path not in sys.path:
    sys.path.append(path)

# Set environment variables
os.environ['PYTHONPATH'] = path

# Import the FastAPI app
from app import app as application