import json
import base64
from mangum import Mangum
from app import app

# Create a Mangum adapter for the FastAPI app
handler = Mangum(app)

# This is the Lambda handler function
# It uses Mangum to convert API Gateway events to ASGI requests
# and ASGI responses to API Gateway responses
