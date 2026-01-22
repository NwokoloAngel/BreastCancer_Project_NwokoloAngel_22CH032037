FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port (standard for documentation, though Render/Heroku ignore this and inject PORT env)
EXPOSE 5000

# Use gunicorn to serve the app
# Bind to 0.0.0.0 and use the PORT environment variable (default 5000)
CMD exec gunicorn --bind 0.0.0.0:$PORT app:app
