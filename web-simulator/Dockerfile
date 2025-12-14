# Meshtastic Web Simulator Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone upstream Meshtasticator (for lib modules)
RUN git clone --depth 1 https://github.com/meshtastic/Meshtasticator.git /app/upstream-meshtasticator

# Copy application files
COPY . .

# Create output directory
RUN mkdir -p /app/upstream-meshtasticator/out

# Expose port
EXPOSE 4000

# Environment variables
ENV PORT=4000
ENV DEBUG=false
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]
