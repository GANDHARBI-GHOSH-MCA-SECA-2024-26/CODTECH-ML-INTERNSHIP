# CODTECH Machine Learning Internship

## Overview

This repository contains two Machine Learning projects completed as part of my CODTECH internship.

## Projects

### 1. Stock Price Predictor

A Machine Learning model that predicts stock prices using historical market data.

**Algorithm:** Linear Regression

**Technologies:**
- Python
- Pandas
- Scikit-learn
- yFinance
- Matplotlib

**Results:**
- Mean Squared Error: 19.76
- R² Score: 0.9694

The model downloads historical Apple stock data, creates features from previous prices, trains a Linear Regression model, evaluates its performance, and predicts the next stock price.

---

### 2. Customer Churn Prediction

A Machine Learning classification model that predicts whether a customer is likely to leave a service.

**Algorithm:** Logistic Regression

**Technologies:**
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

**Example Result:**
- Prediction: Customer likely to churn
- Churn Probability: 76.36%

The project performs data preprocessing, feature scaling, classification, model evaluation, and customer churn prediction.

---

## Project Structure

```text
CODTECH-ML-INTERNSHIP/
├── stock_price_predictor.py
├── customer_churn_prediction.py
├── README.md
└── .gitignore
