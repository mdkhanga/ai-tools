#!/usr/bin/python3

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load synthetic dataset
df = pd.read_csv("synthetic_refunds.csv")

X = df.drop("refund_days", axis=1)
y = df["refund_days"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to DMatrix (optimized for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Training parameters
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42
}

# Train model
print("🚀 Training XGBoost model...")
model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, "test")])

# Evaluate
y_pred = model.predict(dtest)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse = mean_squared_error(y_test, y_pred) ** 0.5


print(f"✅ Model trained. Test RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, "refund_predictor.pkl")
print("✅ Model saved as refund_predictor.pkl")
