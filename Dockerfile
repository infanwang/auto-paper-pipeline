FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data reports docs pdfs

# Set environment variables
ENV PYTHONPATH=/app/scripts
ENV PYTHONUNBUFFERED=1

# Expose port (for web interface)
EXPOSE 8000

# Default command
CMD ["python", "scripts/run_pipeline.py", "--mode", "daily"]
