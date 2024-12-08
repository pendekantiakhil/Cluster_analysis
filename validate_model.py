import pandas as pd
import joblib
import boto3
from sklearn.metrics import accuracy_score

# Load validation dataset
validation_data = pd.read_csv("datasets/ValidationDataset.csv")
X_val = validation_data.drop("target", axis=1)
y_val = validation_data["target"]

# Download the model from S3
s3 = boto3.client("s3")
bucket_name = "<YOUR_S3_BUCKET_NAME>"
s3.download_file(bucket_name, "decision_tree_model.pkl", "decision_tree_model.pkl")

# Load the model
model = joblib.load("decision_tree_model.pkl")

# Make predictions and evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")
