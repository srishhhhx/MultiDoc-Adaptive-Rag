FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 API_HOST=0.0.0.0

# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY run_api.py .

EXPOSE 8000
CMD ["python", "run_api.py"]
