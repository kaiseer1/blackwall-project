FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpcap-dev \
    iptables \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements_new.txt .
RUN pip install --no-cache-dir -r requirements_new.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY datasets/ ./datasets/
COPY models/ ./models/
COPY blackwall_new.py .

# Create necessary directories
RUN mkdir -p logs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8080

# Run as non-root user (optional, comment out if you need packet capture)
# RUN useradd -m -u 1000 blackwall && chown -R blackwall:blackwall /app
# USER blackwall

# Default command
CMD ["python", "blackwall_new.py", "--monitor", "--api"]
