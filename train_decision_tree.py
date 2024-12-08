import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib  # For saving/loading the model
import boto3  # For interacting with AWS S3

# Define S3 bucket details
bucket_name = "<YOUR_S3_BUCKET_NAME>"  # Replace with your bucket name
model_file_name = "decision_tree_model.pkl"

# Load training dataset
train_data = pd.read_csv("datasets/TrainingDataset.csv")
X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model locally
joblib.dump(model, model_file_name)
print(f"Model saved locally as {model_file_name}")

# Upload the model to S3
s3 = boto3.client("s3")
s3.upload_file(model_file_name, bucket_name, model_file_name)
print(f"Model uploaded to S3 bucket '{bucket_name}' as '{model_file_name}'")
