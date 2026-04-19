FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for RDKit
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install torch (CPU-only) separately for smaller image
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source and static files
COPY src/ ./src/
COPY static/ ./static/
COPY weights/ ./weights/
COPY data/ ./data/

# Expose port
EXPOSE 7860

# Run with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
