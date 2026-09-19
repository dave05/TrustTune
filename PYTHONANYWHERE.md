# Deploying TrustTune to PythonAnywhere

This guide provides step-by-step instructions for deploying TrustTune to PythonAnywhere.

## Prerequisites

1. A PythonAnywhere account (Free tier is sufficient for testing)
2. Git installed on your local machine
3. Your TrustTune repository on GitHub

## Setup Steps

### 1. Create a Web App on PythonAnywhere

1. Log in to your PythonAnywhere account
2. Go to the Web tab
3. Click "Add a new web app"
4. Choose "Manual configuration"
5. Select Python 3.10
6. Enter your domain name (e.g., yourusername.pythonanywhere.com)

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
   pip install python-dotenv
   ```

### 4. Configure Environment Variables

1. Create a `.env` file in your project directory:
   ```bash
   nano .env
   ```

2. Add the following environment variables (replace with your actual values):
   ```
   PYTHONANYWHERE_USERNAME=your_username
   DEBUG=False
   SECRET_KEY=your_secret_key
   ALLOWED_HOSTS=your_domain.pythonanywhere.com,localhost,127.0.0.1
   ```

### 5. Configure the WSGI File

1. Go to the Web tab
2. Click on the WSGI configuration file link (e.g., `/var/www/yourusername_pythonanywhere_com_wsgi.py`)
3. Replace the content with the content from your `pythonanywhere_wsgi.py` file
4. Click Save

### 6. Configure Static Files

1. Go to the Web tab
2. In the "Static files" section, add:
   - URL: `/static/`
   - Directory: `/home/yourusername/TrustTune/static`

### 7. Reload the Web App

1. Go to the Web tab
2. Click the "Reload" button for your web app

### 8. Access Your Application

Your application should now be available at:
```
https://yourusername.pythonanywhere.com
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

## Troubleshooting

If you encounter issues:

1. Check the error logs in the Web tab
2. Ensure all dependencies are installed
3. Verify that your WSGI file is correctly configured
4. Make sure your environment variables are set correctly

## Security Notes

- Never commit your `.env` file to version control
- Use a strong, unique SECRET_KEY
- Consider using PythonAnywhere's "secure" option for your web app
- Regularly update your dependencies to patch security vulnerabilities
