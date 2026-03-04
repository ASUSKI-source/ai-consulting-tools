FROM python:3.11-slim

# Base working directory inside the container
WORKDIR /app

# Install dependencies for the Week 4 app
COPY week-4/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the Week 4 application code into the image
COPY week-4/ ./week-4

# Switch to the Week 4 app directory so imports and static paths resolve correctly
WORKDIR /app/week-4

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
