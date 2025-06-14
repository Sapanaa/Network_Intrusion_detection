# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (for scapy and iptables)
RUN apt-get update && apt-get install -y \
    python3-scapy \
    iptables \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY app.py .
COPY src/ src/

# Expose the port for the Streamlit dashboard
EXPOSE 8501

# Environment variable for logging
ENV PYTHONUNBUFFERED=1

# Default command (will be overridden by docker-compose)
CMD ["python", "src/main.py"]