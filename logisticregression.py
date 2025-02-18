import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Generating a synthetic dataset for binary classification
np.random.seed(42)
data = {
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100),
    'Label': np.random.randint(0, 2, 100)  # 0 or 1 for binary classification
}
dataset = pd.DataFrame(data)

# Splitting the dataset into features and target variable
X = dataset[['Feature1', 'Feature2']].values
y = dataset['Label'].values

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression function implementation
def logistic_regression_mds(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Gradient Descent
    for _ in range(iterations):
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        # Compute gradients
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

# Function to make predictions
def predict(X, weights, bias):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return np.array([1 if i > 0.5 else 0 for i in y_pred])

# Training the logistic regression model
weights, bias = logistic_regression_mds(X_train_scaled, y_train)

# Making predictions on the test set
y_pred_custom = predict(X_test_scaled, weights, bias)

# Evaluating the custom model
accuracy_custom = accuracy_score(y_test, y_pred_custom)
conf_matrix_custom = confusion_matrix(y_test, y_pred_custom)
f1_custom = f1_score(y_test, y_pred_custom)

# Printing results
print(f"\nCustom Logistic Regression Model Accuracy: {accuracy_custom * 100:.2f}%")
print(f"Custom Confusion Matrix:\n{conf_matrix_custom}")
print(f"Custom F1 Score: {f1_custom:.2f}")
