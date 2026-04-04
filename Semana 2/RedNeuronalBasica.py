import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(7)

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("Train:", X_train_s.shape, "Test:", X_test_s.shape)

from sklearn.linear_model import LogisticRegression
baseline = LogisticRegression(max_iter=2000, random_state=7)
baseline.fit(X_train_s, y_train)
pred_b = baseline.predict(X_test_s)
print("Accuracy (baseline):", accuracy_score(y_test, pred_b))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_b))

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="adam", max_iter=2000, random_state=7)
mlp.fit(X_train_s, y_train)
pred = mlp.predict(X_test_s)
print("Accuracy (MLP):", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("\nReporte:\n", classification_report(y_test, pred, target_names=data.target_names))