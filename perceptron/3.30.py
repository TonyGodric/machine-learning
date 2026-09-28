
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.linear_model import Perceptron as SkPerceptron


class Perceptron:


    def __init__(self, learning_rate=0.01, n_epochs=100, random_state=42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        self.weights_ = rng.normal(0, 0.01, n_features)
        self.bias_ = 0.0
        self.errors_ = []

        for _ in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X, y):
                y_pred = self._activation(xi)
                update = self.learning_rate * (target - y_pred)
                if update != 0.0:
                    self.weights_ += update * xi
                    self.bias_ += update
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:      # converged: perfectly separates the training data
                break
        return self

    def _net_input(self, x):
        return np.dot(x, self.weights_) + self.bias_

    def _activation(self, x):
        return 1 if self._net_input(x) >= 0 else 0

    def predict(self, X):
        return np.array([self._activation(xi) for xi in X])


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    df = pd.read_csv("transactions_fraud_dataset.csv")
    X = df.drop(columns=["is_fraud"]).values
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- Hyperparameter tuning via 5-fold CV, optimizing F1 ----
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    epoch_options = [50, 100, 200]
    best_f1, best_params = -1, None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for lr in learning_rates:
        for n_epochs in epoch_options:
            fold_f1s = []
            for tr_idx, val_idx in skf.split(X_train_s, y_train):
                model = Perceptron(learning_rate=lr, n_epochs=n_epochs)
                model.fit(X_train_s[tr_idx], y_train[tr_idx])
                preds = model.predict(X_train_s[val_idx])
                fold_f1s.append(f1_score(y_train[val_idx], preds, zero_division=0))
            mean_f1 = float(np.mean(fold_f1s))
            if mean_f1 > best_f1:
                best_f1, best_params = mean_f1, (lr, n_epochs)

    print(f"Best hyperparameters : learning_rate={best_params[0]}, n_epochs={best_params[1]}")
    print(f"Best CV F1-score     : {best_f1:.4f}")

    # ---- Retrain on the full training set, evaluate on the test set ----
    final_model = Perceptron(learning_rate=best_params[0], n_epochs=best_params[1])
    final_model.fit(X_train_s, y_train)
    y_pred = final_model.predict(X_test_s)

    metrics = evaluate(y_test, y_pred)
    print("\n=== From-scratch Perceptron | Test set performance ===")
    for k, v in metrics.items():
        print(f"{k.capitalize():10s}: {v:.4f}")
    print("\nConfusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"], zero_division=0))

    # ---- Bonus: scikit-learn's Perceptron, for a sanity check ----
    sk_model = SkPerceptron(max_iter=1000, eta0=best_params[0],
                             class_weight="balanced", random_state=42)
    sk_model.fit(X_train_s, y_train)
    sk_pred = sk_model.predict(X_test_s)
    sk_metrics = evaluate(y_test, sk_pred)
    print("\n=== scikit-learn Perceptron (class_weight='balanced') | Test set performance ===")
    for k, v in sk_metrics.items():
        print(f"{k.capitalize():10s}: {v:.4f}")


if __name__ == "__main__":
    main()