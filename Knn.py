# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
df = pd.read_csv("iris.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# -----------------------------
# Step 2: Separate Features & Target
# -----------------------------
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# -----------------------------
# Step 3: Convert labels into numbers
# -----------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# -----------------------------
# Step 4: Split the dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Step 5: Normalize the data
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Step 6: Find the best value of K
# -----------------------------
accuracy = []

print("\nAccuracy for Different K Values\n")

for k in range(1, 11):

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = accuracy_score(y_test, predictions)

    accuracy.append(score)

    print(f"K = {k}  Accuracy = {score:.4f}")

# -----------------------------
# Step 7: Best K
# -----------------------------
best_k = accuracy.index(max(accuracy)) + 1

print("\nBest K Value:", best_k)

# -----------------------------
# Step 8: Train Final Model
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# -----------------------------
# Step 9: Evaluation
# -----------------------------
print("\nFinal Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Step 10: Accuracy vs K Graph
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(range(1,11), accuracy, marker='o')

plt.title("Accuracy vs K")

plt.xlabel("K Value")

plt.ylabel("Accuracy")

plt.grid(True)

plt.show()

# -----------------------------
# Step 11: Decision Boundary
# -----------------------------
# Decision boundary works only with 2 features,
# so we use the first two columns.

X = df.iloc[:, :2]

y = encoder.fit_transform(df.iloc[:, -1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train, y_train)

X1, X2 = np.meshgrid(
    np.arange(X_train[:,0].min()-1,
              X_train[:,0].max()+1,
              0.02),

    np.arange(X_train[:,1].min()-1,
              X_train[:,1].max()+1,
              0.02)
)

plt.figure(figsize=(8,6))

plt.contourf(
    X1,
    X2,
    knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.4,
    cmap=ListedColormap(("red","green","blue"))
)

plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=y_train,
    edgecolor="black",
    cmap=ListedColormap(("red","green","blue"))
)

plt.title("KNN Decision Boundary")

plt.xlabel("Sepal Length")

plt.ylabel("Sepal Width")

plt.show()