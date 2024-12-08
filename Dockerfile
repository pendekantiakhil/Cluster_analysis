# Use Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install pandas joblib boto3 scikit-learn

# Copy the validation script
COPY validate_model.py /app/validate_model.py

# Set the command to run the validation script
CMD ["python", "validate_model.py"]
