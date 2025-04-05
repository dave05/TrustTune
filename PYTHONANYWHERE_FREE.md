# Deploying TrustTune to PythonAnywhere (Free Tier)

This guide provides step-by-step instructions for deploying TrustTune to PythonAnywhere's free tier.

## Prerequisites

1. A PythonAnywhere account (Free tier is sufficient)
   - Sign up at [PythonAnywhere](https://www.pythonanywhere.com/)
2. Git installed on your local machine
3. Your TrustTune repository on GitHub

## Setup Steps

### 1. Create a Web App on PythonAnywhere

1. Log in to your PythonAnywhere account (username: dawitBeshah)
2. Go to the Web tab
3. Click "Add a new web app"
4. Choose "Manual configuration"
5. Select Python 3.10
6. Enter your domain name (e.g., dawitbeshah.pythonanywhere.com)

### 2. Clone the Repository

1. Go to the Consoles tab
2. Start a new Bash console
3. Clone your repository:
   ```bash
   git clone https://github.com/dave05/TrustTune.git
   cd TrustTune
   ```

### 3. Set Up a Virtual Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 4. Configure the WSGI File

1. Go to the Web tab
2. Click on the WSGI configuration file link (e.g., `/var/www/dawitbeshah_pythonanywhere_com_wsgi.py`)
3. Replace the content with the content from your `pythonanywhere_wsgi.py` file
4. Click Save

### 5. Configure Static Files

1. Go to the Web tab
2. In the "Static files" section, add:
   - URL: `/static/`
   - Directory: `/home/dawitBeshah/TrustTune/static`

### 6. Set Up CORS for API Access

Since your frontend will be hosted on GitHub Pages and your backend on PythonAnywhere, you need to configure CORS:

1. Make sure your app.py has the CORS middleware configured:
   ```python
   from fastapi.middleware.cors import CORSMiddleware

   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://dave05.github.io", "http://localhost:8000"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

### 7. Reload the Web App

1. Go to the Web tab
2. Click the "Reload" button for your web app

### 8. Access Your Application

Your backend API will be available at:
```
https://dawitbeshah.pythonanywhere.com
```

Your frontend will be available at:
```
https://dave05.github.io/TrustTune
```

## Updating Your Application

To update your application after making changes:

1. Push your changes to GitHub
2. Connect to PythonAnywhere via a Bash console
3. Navigate to your project directory:
   ```bash
   cd ~/TrustTune
   ```
4. Pull the latest changes:
   ```bash
   git pull
   ```
5. Reload your web app from the Web tab

## Free Tier Limitations

PythonAnywhere's free tier has some limitations:

1. Your app will be hosted at `yourusername.pythonanywhere.com` (no custom domains)
2. CPU time and bandwidth are limited
3. Your web app will go to sleep after a period of inactivity
4. Outbound HTTP requests are limited to specific whitelisted domains

## Connecting Frontend to Backend

Update your frontend code to point to your PythonAnywhere backend:

1. In your static GitHub Pages site, update API calls to use:
   ```javascript
   const API_URL = "https://dawitbeshah.pythonanywhere.com";
   ```

2. For API examples in documentation, use:
   ```
   https://dawitbeshah.pythonanywhere.com/calibrate
   ```

## Troubleshooting

If you encounter issues:

1. Check the error logs in the Web tab
2. Ensure all dependencies are installed
3. Verify that your WSGI file is correctly configured
4. Check for CORS issues in your browser's developer console
