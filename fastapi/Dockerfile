FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app files into /app
COPY . .

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]