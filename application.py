import os
from app import app as application

# This file is used by AWS Elastic Beanstalk to run your application
# The variable 'application' is required by Elastic Beanstalk

if __name__ == "__main__":
    # For local testing
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(application, host="0.0.0.0", port=port)
