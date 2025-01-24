
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Create a pandas DataFrame from the Iris data
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add a 'target' column to the DataFrame with the target labels (species)
df['target'] = iris.target

# Display the first few rows of the DataFrame
print(df.head())
print(df['target'])

# Separate data for each species
setosa_data = df[df['target'] == 0]
virginica_data = df[df['target'] == 2]
versicolor_data = df[df['target'] == 1]

# Randomly select 10 samples each for Setosa and Virginica for training
train_setosa = setosa_data.sample(n=10, random_state=42)
train_virginica = virginica_data.sample(n=10, random_state=42)

# Combine the training samples into a single DataFrame
train_df = pd.concat([train_setosa, train_virginica])

# Separate features (X_train) and target labels (y_train) for training data
X_train = train_df.drop('target', axis=1).values
y_train = train_df['target'].values

# Randomly select 3 Setosa, 3 Virginica, and 2 Versicolor for testing
test_setosa = setosa_data.drop(train_setosa.index).sample(n=3, random_state=42)
test_virginica = virginica_data.drop(train_virginica.index).sample(n=3, random_state=42)
test_versicolor = versicolor_data.sample(n=2, random_state=42)

# Combine the test samples into a single DataFrame
test_df = pd.concat([test_setosa, test_virginica, test_versicolor])

# Separate features (X_test) and target labels (y_test) for testing data
X_test = test_df.drop('target', axis=1).values
y_test = test_df['target'].values

# Define the Euclidean distance function
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Implement the KNN algorithm
def knn_predict(X_train, y_train, X_test, k):
    y_pred = []
    for x in X_test:
        distances = []
        for i, x_train in enumerate(X_train):
            distance = euclidean_distance(x, x_train)
            distances.append((distance, y_train[i]))
            print(f"Distance: {distance:.4f}")

        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        k_nearest_labels = [label for (_, label) in k_nearest_neighbors]
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        y_pred.append(most_common)
    return np.array(y_pred)

# Evaluate the model
def evaluate_model(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return cm, accuracy, precision, recall, f1

# Run the evaluation for different values of K
ks = range(1, 6)

for k in ks:
    print(f"\nEvaluating K={k}")
    y_pred = knn_predict(X_train, y_train, X_test, k)
    cm, accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

    print(f"K={k}, Distance Metric=Euclidean")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("\n")

# --- Visualizing the Performance Metrics ---
import matplotlib.pyplot as plt

metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_values = [accuracy, precision, recall, f1]

# Create a plot for metrics
plt.figure(figsize=(8, 6))
plt.plot(metrics_labels, metric_values, marker='o', linestyle='-', color='green')
plt.ylim(0, 1.1)
plt.title('KNN Evaluation Metrics')
plt.ylabel('Metric Score')
plt.show()
