import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import joblib
import boto3

# Define GitHub raw file URLs
training_data_url = "https://raw.githubusercontent.com/pendekantakhil/Cluster_analysis/main/datasets/TrainingDataset.csv"
validation_data_url = "https://raw.githubusercontent.com/pendekantakhil/Cluster_analysis/main/datasets/ValidationDataset%20(2).csv"

# Load datasets directly from GitHub
train_data = pd.read_csv(training_data_url)
validation_data = pd.read_csv(validation_data_url)

# Prepare training data
X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model locally
model_file_name = "decision_tree_model.pkl"
joblib.dump(model, model_file_name)
print(f"Model saved locally as {model_file_name}")

# Upload the model to S3
s3 = boto3.client("s3")
bucket_name = "<cloudclusteraws>"  # Replace with your S3 bucket name
s3.upload_file(model_file_name, bucket_name, model_file_name)
print(f"Model uploaded to S3 bucket '{bucket_name}' as '{model_file_name}'")
