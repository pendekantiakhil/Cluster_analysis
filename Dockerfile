# Use Python 3.9 slim as the base image
FROM python:3.9-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install necessary Python packages
RUN pip install --no-cache-dir pandas scikit-learn boto3 requests

# Set the working directory
WORKDIR /app

# Copy the application file to the container
COPY app.py .

# Command to run the application
CMD ["python", "app.py"]
