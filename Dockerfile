FROM python:3.11-slim

WORKDIR /app

COPY week-3/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY week-3/ .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
