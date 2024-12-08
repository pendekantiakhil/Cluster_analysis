import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import boto3

# Load the training dataset
training_data = pd.read_csv("datasets/TrainingDataset.csv")
X = training_data.drop("target", axis=1)
y = training_data["target"]

# Train the decision tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model locally
joblib.dump(model, "decision_tree_model.pkl")

# Upload the model to S3
s3 = boto3.client("s3")
bucket_name = "<cloudclusteraws>"
s3.upload_file("decision_tree_model.pkl", bucket_name, "decision_tree_model.pkl")

print("Model uploaded to S3 successfully!")

