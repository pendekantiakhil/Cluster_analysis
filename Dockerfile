FROM python:3.9-slim

# Install dependencies
RUN pip install boto3 pandas scikit-learn joblib

# Copy the validation script and datasets
COPY validate_model.py /app/validate_model.py
COPY datasets/ValidationDataset.csv /app/ValidationDataset.csv

# Set the working directory
WORKDIR /app

# Run the script
CMD ["python", "validate_model.py"]
