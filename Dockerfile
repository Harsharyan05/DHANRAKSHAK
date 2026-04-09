FROM python:3.12-slim

# Avoid stdout buffering
ENV PYTHONUNBUFFERED=1

# Install OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Workdir inside container
WORKDIR /code

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full app
COPY . .

# Expose HuggingFace port
EXPOSE 7860

# Run FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
