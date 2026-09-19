#!/bin/bash
# Script to deploy TrustTune to PythonAnywhere

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Configuration
PA_USERNAME="$PYTHONANYWHERE_USERNAME"
PA_DOMAIN="$PA_USERNAME.pythonanywhere.com"
APP_PATH="/home/$PA_USERNAME/TrustTune"
PA_API_TOKEN="$PYTHONANYWHERE_API_TOKEN"

echo "Deploying TrustTune to PythonAnywhere..."

# 1. Push changes to GitHub
echo "Pushing changes to GitHub..."
git add .
git commit -m "Update for deployment"
git push origin main

# 2. Connect to PythonAnywhere and update the code
echo "Connecting to PythonAnywhere and updating code..."
ssh $PA_USERNAME@ssh.pythonanywhere.com << EOF
    cd $APP_PATH
    git pull
    source venv/bin/activate
    pip install -r requirements.txt
    touch $APP_PATH/pythonanywhere_wsgi.py
EOF

# 3. Reload the web app
echo "Reloading the web app..."
curl -X POST \
    -H "Authorization: Token $PA_API_TOKEN" \
    https://www.pythonanywhere.com/api/v0/user/$PA_USERNAME/webapps/$PA_DOMAIN/reload/

echo "Deployment complete!"
echo "Your application is available at: https://$PA_DOMAIN"
