# Customer Churn Prediction
# Install: pip install pandas numpy scikit-learn matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Reproducible dataset
np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "Age": np.random.randint(18, 70, n),
    "MonthlyCharges": np.random.randint(20, 150, n),
    "Tenure": np.random.randint(1, 72, n),
    "SupportCalls": np.random.randint(0, 10, n),
    "ContractLength": np.random.choice([1, 12, 24], n),
    "InternetService": np.random.choice([0, 1], n)
})

# Generate realistic churn patterns
churn_probability = (
    0.15
    + 0.002 * data["MonthlyCharges"]
    - 0.004 * data["Tenure"]
    + 0.04 * data["SupportCalls"]
    - 0.001 * data["ContractLength"]
    + 0.08 * data["InternetService"]
)

churn_probability = np.clip(churn_probability, 0.05, 0.95)

data["Churn"] = np.random.binomial(1, churn_probability)

print("===== CUSTOMER CHURN PREDICTION =====")
print("\nDataset:")
print(data.head())

# Features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Predict a new customer
new_customer = pd.DataFrame({
    "Age": [25],
    "MonthlyCharges": [120],
    "Tenure": [5],
    "SupportCalls": [7],
    "ContractLength": [1],
    "InternetService": [1]
})

new_customer_scaled = scaler.transform(new_customer)

result = model.predict(new_customer_scaled)[0]
probability = model.predict_proba(new_customer_scaled)[0][1]

print("\n===== NEW CUSTOMER PREDICTION =====")

if result == 1:
    print("Prediction: CUSTOMER LIKELY TO CHURN")
else:
    print("Prediction: CUSTOMER LIKELY TO STAY")

print("Churn Probability:", round(probability * 100, 2), "%")

# Visualization
data["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = Stay, 1 = Churn)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()