

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# PHASE 1: INPUT - load and understand the dataset
# ---------------------------------------------------------------
iris = load_iris()
X = iris.data            # 4 features: sepal length/width, petal length/width (cm)
y = iris.target          # 3 classes: setosa, versicolor, virginica

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(y, iris.target_names)

print("=== DATASET OVERVIEW ===")
print(f"Samples: {X.shape[0]} | Features: {X.shape[1]} | Classes: {len(iris.target_names)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df["species"].value_counts())
print("\nMissing values:", int(df.isnull().sum().sum()))
print("\nSummary statistics:")
print(df.describe().round(2))

# ---------------------------------------------------------------
# PHASE 2: PROCESS - split, scale, train
# ---------------------------------------------------------------
# Shuffle + 80/20 split. stratify keeps class balance in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# Scale: fit ONLY on training data to avoid data leakage, then transform both.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose K: try several values, keep the best one by cross-validation on the
# training set (the test set stays untouched until the final evaluation).
from sklearn.model_selection import cross_val_score

k_values = range(1, 21)
cv_scores = [
    cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_scaled, y_train, cv=5).mean()
    for k in k_values
]
best_k = k_values[int(np.argmax(cv_scores))]
print(f"\nBest K from 5-fold cross-validation: {best_k}")

# Instantiate -> Fit -> Predict
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# ---------------------------------------------------------------
# PHASE 3: OUTPUT - validate the model
# ---------------------------------------------------------------
print("\n=== MODEL EVALUATION ===")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(f"Macro F1 score: {f1_score(y_test, predictions, average='macro'):.3f}")

print("\nConfusion Matrix (rows = actual, columns = predicted):")
cm = confusion_matrix(y_test, predictions)
print(cm)

print("\nClassification Report (precision / recall / F1 per class):")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# ---------------------------------------------------------------
# Charts (saved as images you can add to your report)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix")

axes[1].plot(list(k_values), [1 - s for s in cv_scores], marker="o")
axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
axes[1].set_xlabel("K (number of neighbors)")
axes[1].set_ylabel("Error rate (cross-validation)")
axes[1].set_title("Choosing K")
axes[1].legend()

plt.tight_layout()
plt.savefig("iris_results.png", dpi=150)
print("\nSaved chart: iris_results.png")
plt.show()

# ---------------------------------------------------------------
# Bonus: predict a brand-new flower
# ---------------------------------------------------------------
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])           # sepal L, sepal W, petal L, petal W
new_scaled = scaler.transform(new_flower)               # use the SAME scaler
pred = model.predict(new_scaled)[0]
print(f"\nNew flower {new_flower.tolist()[0]} -> predicted species: {iris.target_names[pred]}")