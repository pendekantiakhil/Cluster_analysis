import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# Load datasets
train_data = pd.read_csv("datasets/TrainingDataset.csv")
validation_data = pd.read_csv("datasets/ValidationDataset.csv")

# Prepare training data
X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]

# Prepare validation data
X_val = validation_data.drop("target", axis=1)
y_val = validation_data["target"]

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate F1 score
f1 = f1_score(y_val, y_pred, average="weighted")  # Use 'weighted' for multi-class
print(f"F1 Score: {f1}")
