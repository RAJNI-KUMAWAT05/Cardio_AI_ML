import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("heart.csv")

# Convert categorical → numeric
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ✅ USE RANDOM FOREST (key change)
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Save columns
joblib.dump(X.columns.tolist(), "model_columns.pkl")

# Save model
joblib.dump(model, "cardio_model.pkl")

print("Unique target values:", y.unique())
print("Sample predictions:", model.predict(X_test[:5]))
print("Sample probabilities:", model.predict_proba(X_test[:5]))
print("✅ Random Forest model trained!")