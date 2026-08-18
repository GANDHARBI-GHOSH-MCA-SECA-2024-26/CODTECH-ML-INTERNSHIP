# Stock Price Predictor
# Install: pip install yfinance scikit-learn matplotlib pandas

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Choose stock
ticker = "AAPL"

# Download historical data
data = yf.download(ticker, period="5y", auto_adjust=True)

# Use closing price
df = data[["Close"]].copy()
df.columns = ["Close"]

# Create prediction features
df["Previous_Day"] = df["Close"].shift(1)
df["Previous_5_Days"] = df["Close"].shift(5)
df["Previous_10_Days"] = df["Close"].shift(10)

df.dropna(inplace=True)

# Features and target
X = df[["Previous_Day", "Previous_5_Days", "Previous_10_Days"]]
y = df["Close"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("===== STOCK PRICE PREDICTOR =====")
print("Stock:", ticker)
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 4))

# Predict next price
latest = df.iloc[-1]

next_day = model.predict([[
    latest["Close"],
    df["Close"].iloc[-5],
    df["Close"].iloc[-10]
]])

print("\nLatest Closing Price:", round(float(latest["Close"]), 2))
print("Predicted Next Price:", round(float(next_day[0]), 2))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Trading Days")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()