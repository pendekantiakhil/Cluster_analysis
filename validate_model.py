import pandas as pd
import joblib
import boto3
from sklearn.metrics import accuracy_score

# Define S3 bucket details
bucket_name = "cloudclusteraws"  # Your S3 bucket name
model_file_name = "decision_tree_model.pkl"
validation_data_url = "https://raw.githubusercontent.com/pendekantiakhil/Cluster_analysis/main/datasets/ValidationDataset%20(2).csv"

# Load validation dataset directly from GitHub
validation_data = pd.read_csv(validation_data_url, sep=";")  # Load with correct delimiter

# Clean up column headers
validation_data.columns = validation_data.columns.str.strip()  # Remove extra spaces
print("Dataset columns:", validation_data.columns)  # Debugging step

# Prepare validation data
X_val = validation_data.drop("quality", axis=1)  # Target variable is 'quality'
y_val = validation_data["quality"]

# Download the model from S3
s3 = boto3.client("s3")
s3.download_file(bucket_name, model_file_name, model_file_name)
print(f"Model downloaded from S3 bucket '{bucket_name}'")

# Load the model
model = joblib.load(model_file_name)
print("Model loaded successfully")

# Make predictions and evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")
