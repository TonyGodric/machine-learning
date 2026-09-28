

import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=100, random_state=42):
        """
        learning_rate : step size eta for weight updates
        n_epochs      : max number of passes over the training data
        random_state  : seed for reproducible weight initialization
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        """
        Train the model.
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,), labels in {0, 1}
        """
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]

        # small random init instead of all-zeros: avoids ties in the
        # dot product when features are symmetric around zero
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0
        self.errors_ = []  # number of misclassifications per epoch, for diagnostics

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
            if errors == 0:       # converged: training data perfectly separated
                break
        return self

    def _net_input(self, x):
        return np.dot(x, self.weights_) + self.bias_

    def _activation(self, x):
        return 1 if self._net_input(x) >= 0 else 0

    def predict(self, X):
        """Predict labels (0 or 1) for new, unseen data X."""
        return np.array([self._activation(xi) for xi in X])


if __name__ == "__main__":
    # Small self-contained demo so you can run this file directly and see
    # that fit()/predict() work as expected.
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    X, y = make_classification(
        n_samples=200, n_features=2, n_redundant=0,
        n_clusters_per_class=1, class_sep=2.0, random_state=1
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = Perceptron(learning_rate=0.1, n_epochs=50, random_state=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Epochs until convergence (or max reached):", len(model.errors_))
    print("Demo accuracy on held-out test set:", accuracy_score(y_test, y_pred))