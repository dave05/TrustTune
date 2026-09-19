# Deploying TrustTune

This guide provides instructions for deploying TrustTune to various hosting platforms.

## PythonAnywhere

PythonAnywhere is a cloud-based Python development and hosting environment that makes it easy to deploy Python web applications.

### Steps to Deploy

1. **Create a PythonAnywhere Account**
   - Sign up at [PythonAnywhere](https://www.pythonanywhere.com/)
   - Choose a plan that suits your needs (Free tier is available for testing)

2. **Set Up a Web App**
   - Go to the Web tab in your PythonAnywhere dashboard
   - Click "Add a new web app"
   - Choose "Manual configuration" and select Python 3.10

3. **Clone the Repository**
   - Open a Bash console from your PythonAnywhere dashboard
   - Clone the repository:
     ```bash
     git clone https://github.com/dave05/TrustTune.git
     cd TrustTune
     ```

4. **Set Up a Virtual Environment**
   - Create and activate a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

5. **Configure the WSGI File**
   - Go to the Web tab and click on the WSGI configuration file link
   - Replace the content with the following:
     ```python
     import os
     import sys

     # Add the project directory to the Python path
     path = '/home/{username}/TrustTune'
     if path not in sys.path:
         sys.path.append(path)

     # Set environment variables
     os.environ['PYTHONPATH'] = path

     # Import the FastAPI app
     from app import app

     # PythonAnywhere WSGI handler
     def application(environ, start_response):
         """
         WSGI application function for PythonAnywhere.
         """
         # Call the FastAPI application
         return app(environ, start_response)
     ```
   - Replace `{username}` with your PythonAnywhere username

6. **Configure Static Files**
   - Go to the Web tab
   - Add the following static file mappings:
     - URL: `/static/` -> Directory: `/home/{username}/TrustTune/static`

7. **Reload the Web App**
   - Click the "Reload" button for your web app

8. **Access Your Application**
   - Your application will be available at `https://{username}.pythonanywhere.com`

## GitHub Codespaces

TrustTune is also configured to run on GitHub Codespaces. Follow these steps:

1. Go to the GitHub repository: https://github.com/dave05/TrustTune
2. Click the "Code" button
3. Select the "Codespaces" tab
4. Click "Create codespace on main"
5. Once the codespace is ready, run:
   ```bash
   ./.devcontainer/codespace-setup.sh && uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
6. The application will be available at the URL shown in the "Ports" tab

## Heroku

To deploy to Heroku:

1. **Create a Heroku Account**
   - Sign up at [Heroku](https://www.heroku.com/)

2. **Install the Heroku CLI**
   - Follow the instructions at [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)

3. **Login to Heroku**
   ```bash
   heroku login
   ```

4. **Create a Heroku App**
   ```bash
   heroku create trusttune-app
   ```

5. **Add a Procfile**
   - Create a file named `Procfile` in the root directory with the following content:
     ```
     web: uvicorn app:app --host=0.0.0.0 --port=${PORT:-8000}
     ```

6. **Deploy to Heroku**
   ```bash
   git push heroku main
   ```

7. **Open the App**
   ```bash
   heroku open
   ```

## Docker

To deploy using Docker:

1. **Build the Docker Image**
   ```bash
   docker build -t trusttune .
   ```

2. **Run the Container**
   ```bash
   docker run -p 8000:8000 trusttune
   ```

3. **Access the Application**
   - Open your browser and go to `http://localhost:8000`
