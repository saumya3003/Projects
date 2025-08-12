# public_health_ml_pipeline.py

"""
Supervised Learning Project on Public Health Data

Built end-to-end ML pipelines using Python to process vaccination data and apply
supervised learning (linear and logistic regression).
Tools & Technologies: Python, Pandas, NumPy, scikit-learn, Jupyter Notebook
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# Load vaccination dataset (replace with your own CSV path)
df = pd.read_csv('vaccination_data.csv')
print("Data Preview:")
print(df.head())

# Example preprocessing (handle missing values, encode categories)
df = df.dropna()

# Assume numerical features and target for regression
y_reg = df['vaccination_rate']
X_reg = df.drop(columns=['vaccination_rate', 'high_risk_flag'])

# Train/test split for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Linear Regression model
lin_model = LinearRegression()
lin_model.fit(X_train_reg, y_train_reg)
y_pred_reg = lin_model.predict(X_test_reg)
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))

# Logistic Regression example (predicting high_risk_flag)
y_clf = df['high_risk_flag']  # 1 = high risk, 0 = not high risk
X_clf = df.drop(columns=['high_risk_flag', 'vaccination_rate'])

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_clf, y_train_clf)
y_pred_clf = log_model.predict(X_test_clf)
print("Logistic Regression Accuracy:", accuracy_score(y_test_clf, y_pred_clf))

# Save models if needed
import joblib
joblib.dump(lin_model, 'linear_regression_model.pkl')
joblib.dump(log_model, 'logistic_regression_model.pkl')

"""
Instructions:
1. Place your vaccination_data.csv in the same directory.
2. Ensure it has numeric features + 'vaccination_rate' (for regression) and 'high_risk_flag' (for classification).
3. Run this script in Jupyter Notebook or Python.
4. Outputs: RMSE for regression, accuracy for classification, and saved models.
"""
