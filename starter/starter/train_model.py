# Script to train machine learning model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

from ml.data import process_data
from ml.model import compute_model_metrics

# Load data
data = pd.read_csv("../data/census.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Print columns to verify
print("Columns:")
print(data.columns)

# Split data
train, test = train_test_split(data, test_size=0.20, random_state=42)

# Categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Process testing data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("\nModel Metrics:")
print("Precision:", precision)
print("Recall:", recall)
print("Fbeta:", fbeta)

# Save model and encoders
joblib.dump(model, "../model/model.pkl")
joblib.dump(encoder, "../model/encoder.pkl")
joblib.dump(lb, "../model/lb.pkl")

print("\nModel saved successfully!")
