"""Demonstrate overfitting and correct it with a validation set.

The final test set is evaluated only once, after the best epoch has been
selected using the validation set.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent


def load_dataset(filename: str) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(DATA_DIR / filename)
    return data["area"].to_numpy(dtype=float), data["price"].to_numpy(dtype=float)


def make_features(
    area: np.ndarray, area_mean: float, area_std: float, degree: int
) -> np.ndarray:
    """Create scaled polynomial features, including the bias column."""
    # Keep higher powers numerically stable while retaining the train scaling.
    scaled_area = (area - area_mean) / (2.0 * area_std)
    return np.column_stack([scaled_area**power for power in range(degree + 1)])


def predict_price(
    area: np.ndarray, weights: np.ndarray, area_mean: float, area_std: float
) -> np.ndarray:
    features = make_features(area, area_mean, area_std, len(weights) - 1)
    return features @ weights


def mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((actual - predicted) ** 2))


def train_model(
    train_area: np.ndarray,
    train_price: np.ndarray,
    validation_area: np.ndarray,
    validation_price: np.ndarray,
    degree: int,
    learning_rate: float,
    iterations: int,
    patience: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], int]:
    """Train until convergence while retaining the best validation weights.

    The returned final weights intentionally show the overfit model. The best
    weights are the validation-based correction that should be used for test
    predictions.
    """
    area_mean = float(np.mean(train_area))
    area_std = float(np.std(train_area))
    if area_std == 0:
        raise ValueError("Training areas must not all have the same value.")

    train_features = make_features(train_area, area_mean, area_std, degree)
    validation_features = make_features(
        validation_area, area_mean, area_std, degree
    )
    weights = np.zeros(degree + 1, dtype=float)
    best_weights = weights.copy()
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(iterations):
        errors = train_features @ weights - train_price
        gradient = (train_features.T @ errors) / len(train_price)
        weights -= learning_rate * gradient

        train_loss = mean_squared_error(train_price, train_features @ weights)
        validation_loss = mean_squared_error(
            validation_price, validation_features @ weights
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = weights.copy()
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Do not stop: the graph should visibly contain the overfitting region.
        # `best_weights` is used later to correct the overfit model.
        if epochs_without_improvement >= patience and epoch > iterations // 2:
            break

    return (
        weights,
        best_weights,
        train_losses,
        validation_losses,
        best_epoch,
    )


def plot_learning_curves(
    train_losses: list[float],
    validation_losses: list[float],
    best_epoch: int,
    output_path: Path,
) -> None:
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, label="Train MSE")
    plt.plot(epochs, validation_losses, label="Validation MSE")
    overfit_mask = np.asarray(validation_losses) > np.asarray(train_losses)
    plt.fill_between(
        epochs,
        train_losses,
        validation_losses,
        where=overfit_mask,
        color="tab:red",
        alpha=0.15,
        label="Overfit region (validation > train)",
    )
    plt.axvline(
        best_epoch + 1,
        color="tab:green",
        linestyle="--",
        label=f"Best validation epoch ({best_epoch + 1})",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("Overfitting detected and corrected with validation")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate(
    name: str,
    area: np.ndarray,
    price: np.ndarray,
    weights: np.ndarray,
    train_area: np.ndarray,
) -> float:
    prediction = predict_price(
        area,
        weights,
        float(np.mean(train_area)),
        float(np.std(train_area)),
    )
    loss = mean_squared_error(price, prediction)
    print(f"{name} MSE: {loss:.6f}")
    return loss


def main() -> None:
    train_area, train_price = load_dataset("data train.csv")
    validation_area, validation_price = load_dataset("data validation.csv")
    test_area, test_price = load_dataset("data test.csv")

    overfit_weights, corrected_weights, train_losses, validation_losses, best_epoch = (
        train_model(
            train_area,
            train_price,
            validation_area,
            validation_price,
            degree=20,
            learning_rate=0.005,
            iterations=20000,
            patience=2000,
        )
    )

    graph_path = DATA_DIR / "overfitting_validation.png"
    plot_learning_curves(train_losses, validation_losses, best_epoch, graph_path)

    area_mean = float(np.mean(train_area))
    area_std = float(np.std(train_area))
    final_train_loss = mean_squared_error(
        train_price, predict_price(train_area, overfit_weights, area_mean, area_std)
    )
    best_validation_loss = mean_squared_error(
        validation_price,
        predict_price(validation_area, corrected_weights, area_mean, area_std),
    )

    print(f"Overfit model train MSE: {final_train_loss:.6f}")
    print(f"Best validation epoch: {best_epoch + 1}")
    print(f"Corrected validation MSE: {best_validation_loss:.6f}")
    print(f"Graph saved to: {graph_path}")
    evaluate("Corrected test", test_area, test_price, corrected_weights, train_area)


if __name__ == "__main__":
    main()
