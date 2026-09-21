import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Tương thích với cả Matplotlib cũ và mới.
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

pd.set_option("display.float_format", lambda value: f"{value:.4f}")
RANDOM_SEED = 1
SPLIT_SEED = 37
N_SAMPLES = 70

rng = np.random.default_rng(RANDOM_SEED)

area = np.sort(rng.uniform(35, 200, N_SAMPLES))
true_price = 1 + 0.06 * area
observed_price = true_price + rng.normal(0, 0.7, N_SAMPLES)

df = pd.DataFrame({
    "area_m2": area,
    "price_billion_vnd": observed_price,
    "true_price": true_price,
})

df.head(10)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.scatter(
    df["area_m2"],
    df["price_billion_vnd"],
    color="#2563EB",
    alpha=0.75,
    s=48,
    label="Dữ liệu quan sát",
)
ax.plot(
    df["area_m2"],
    df["true_price"],
    "k--",
    linewidth=2.2,
    label="Quan hệ thật",
)
ax.set(
    title="Dữ liệu giá nhà có nhiễu",
    xlabel="Diện tích nhà (m²)",
    ylabel="Giá nhà (tỷ đồng)",
)
ax.legend()
plt.show()
X = df[["area_m2"]].to_numpy()
y = df["price_billion_vnd"].to_numpy()
all_idx = np.arange(N_SAMPLES)

train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.40, random_state=SPLIT_SEED
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=SPLIT_SEED
)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

train_val_idx = np.concatenate([train_idx, val_idx])
X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]

print(f"Train:      {len(train_idx)} mẫu")
print(f"Validation: {len(val_idx)} mẫu")
print(f"Test:       {len(test_idx)} mẫu")
colors = {
    "Train": "#2563EB",
    "Validation": "#F59E0B",
    "Test": "#DC2626",
}

fig, ax = plt.subplots(figsize=(10, 5.5))
for name, indices in [
    ("Train", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx),
]:
    ax.scatter(
        X[indices, 0],
        y[indices],
        s=50,
        alpha=0.78,
        color=colors[name],
        label=name,
    )

ax.plot(area, true_price, "k--", linewidth=2, label="Quan hệ thật")
ax.set(
    title="Phân chia dữ liệu",
    xlabel="Diện tích nhà (m²)",
    ylabel="Giá nhà (tỷ đồng)",
)
ax.legend(ncol=4)
plt.show()
def make_polynomial_model(degree, estimator):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def regression_metrics(model, X_part, y_part):
    prediction = model.predict(X_part)
    return {
        "RMSE": mean_squared_error(y_part, prediction) ** 0.5,
        "MAE": mean_absolute_error(y_part, prediction),
        "R2": r2_score(y_part, prediction),
    }


def evaluate_three_sets(model):
    return pd.DataFrame({
        "Train": regression_metrics(model, X_train, y_train),
        "Validation": regression_metrics(model, X_val, y_val),
        "Test": regression_metrics(model, X_test, y_test),
    }).T


x_grid = np.linspace(30, 205, 700).reshape(-1, 1)
y_grid_true = 1 + 0.06 * x_grid.ravel()
linear_model_train = make_polynomial_model(1, LinearRegression())
linear_model_train.fit(X_train, y_train)

linear_metrics = evaluate_three_sets(linear_model_train)
linear_metrics
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.scatter(X_train[:, 0], y_train, color=colors["Train"], alpha=0.75, label="Train")
ax.scatter(X_val[:, 0], y_val, color=colors["Validation"], alpha=0.75, label="Validation")
ax.scatter(X_test[:, 0], y_test, color=colors["Test"], alpha=0.75, label="Test")
ax.plot(x_grid, y_grid_true, "k--", linewidth=2, label="Quan hệ thật")
ax.plot(
    x_grid,
    linear_model_train.predict(x_grid),
    color="#059669",
    linewidth=2.5,
    label="Hồi quy tuyến tính bậc 1",
)
ax.set(
    title="Mô hình bậc 1 học xu hướng tổng quát",
    xlabel="Diện tích nhà (m²)",
    ylabel="Giá nhà (tỷ đồng)",
)
ax.legend(ncol=3)
plt.show()
overfit_model = make_polynomial_model(15, LinearRegression())
overfit_model.fit(X_train, y_train)

overfit_metrics = evaluate_three_sets(overfit_model)
overfit_metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)

for ax, title, model in [
    (axes[0], "Hồi quy tuyến tính bậc 1", linear_model_train),
    (axes[1], "Hồi quy đa thức bậc 15", overfit_model),
]:
    ax.scatter(X_train[:, 0], y_train, color=colors["Train"], alpha=0.72, label="Train")
    ax.scatter(X_val[:, 0], y_val, color=colors["Validation"], alpha=0.72, label="Validation")
    ax.scatter(X_test[:, 0], y_test, color=colors["Test"], alpha=0.72, label="Test")
    ax.plot(x_grid, y_grid_true, "k--", linewidth=1.8, label="Quan hệ thật")
    ax.plot(x_grid, model.predict(x_grid), color="#7C3AED", linewidth=2.3, label="Dự báo")
    ax.set(title=title, xlabel="Diện tích (m²)", ylabel="Giá (tỷ đồng)")
    ax.legend()

axes[1].set_ylim(0, 15)
fig.suptitle("Mô hình bậc cao khớp cả nhiễu của dữ liệu", fontsize=15, y=1.02)
fig.tight_layout()
plt.show()
degrees = list(range(1, 16))
train_rmse_by_degree = []
val_rmse_by_degree = []

for degree in degrees:
    model = make_polynomial_model(degree, LinearRegression())
    model.fit(X_train, y_train)
    train_rmse_by_degree.append(
        mean_squared_error(y_train, model.predict(X_train)) ** 0.5
    )
    val_rmse_by_degree.append(
        mean_squared_error(y_val, model.predict(X_val)) ** 0.5
    )

best_degree = degrees[int(np.argmin(val_rmse_by_degree))]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(degrees, train_rmse_by_degree, marker="o", color="#2563EB", label="Train RMSE")
ax.plot(degrees, val_rmse_by_degree, marker="o", color="#F59E0B", label="Validation RMSE")
ax.axvline(
    best_degree,
    color="#059669",
    linestyle="--",
    label=f"Bậc tốt nhất theo Validation = {best_degree}",
)
ax.set_xticks(degrees)
ax.set_xticklabels([str(degree) for degree in degrees])
ax.set(
    title="Độ phức tạp tăng làm xuất hiện overfitting",
    xlabel="Bậc đa thức",
    ylabel="RMSE (tỷ đồng)",
)
ax.legend()
plt.show()
simple_model = make_polynomial_model(best_degree, LinearRegression())
simple_model.fit(X_train_val, y_train_val)

simple_test_metrics = regression_metrics(simple_model, X_test, y_test)
pd.DataFrame(simple_test_metrics, index=[f"Degree = {best_degree}"])
cv = KFold(n_splits=5, shuffle=True, random_state=42)

ridge_search = GridSearchCV(
    estimator=make_polynomial_model(15, Ridge()),
    param_grid={"model__alpha": np.logspace(-6, 4, 30)},
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
)
ridge_search.fit(X_train_val, y_train_val)
ridge_model = ridge_search.best_estimator_

print(f"Alpha tốt nhất: {ridge_search.best_params_['model__alpha']:.6g}")
print(f"CV RMSE: {-ridge_search.best_score_:.4f} tỷ đồng")
pd.DataFrame(
    regression_metrics(ridge_model, X_test, y_test),
    index=["Degree 15 + Ridge"],
)
tuned_search = GridSearchCV(
    estimator=make_polynomial_model(1, Ridge()),
    param_grid={
        "poly__degree": list(range(1, 11)),
        "model__alpha": np.logspace(-5, 3, 20),
    },
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
)
tuned_search.fit(X_train_val, y_train_val)
tuned_model = tuned_search.best_estimator_

print("Siêu tham số tốt nhất:", tuned_search.best_params_)
print(f"CV RMSE: {-tuned_search.best_score_:.4f} tỷ đồng")
pd.DataFrame(
    regression_metrics(tuned_model, X_test, y_test),
    index=["Degree + alpha chọn bằng CV"],
)
models = {
    "Bậc 15 không regularization": overfit_model,
    f"Giảm độ phức tạp: bậc {best_degree}": simple_model,
    "Bậc 15 + Ridge": ridge_model,
    "Degree + alpha bằng CV": tuned_model,
}

comparison_rows = []
for name, model in models.items():
    prediction = model.predict(X_test)
    comparison_rows.append({
        "Mô hình": name,
        "Test RMSE": mean_squared_error(y_test, prediction) ** 0.5,
        "Test MAE": mean_absolute_error(y_test, prediction),
        "Test R2": r2_score(y_test, prediction),
    })

comparison = pd.DataFrame(comparison_rows).sort_values("Test RMSE")
comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

for ax, (name, model) in zip(axes.ravel(), models.items()):
    ax.scatter(X_train_val[:, 0], y_train_val, color="#2563EB", alpha=0.60, label="Train + Validation")
    ax.scatter(X_test[:, 0], y_test, color="#DC2626", alpha=0.82, label="Test")
    ax.plot(x_grid, y_grid_true, "k--", linewidth=1.8, label="Quan hệ thật")
    ax.plot(x_grid, model.predict(x_grid), color="#7C3AED", linewidth=2.3, label="Dự báo")
    test_rmse = mean_squared_error(y_test, model.predict(X_test)) ** 0.5
    ax.set_title(f"{name}\nTest RMSE = {test_rmse:.3f}")
    ax.set_xlabel("Diện tích (m²)")
    ax.set_ylabel("Giá (tỷ đồng)")
    ax.set_ylim(0, 15)
    ax.legend()

fig.suptitle("So sánh trước và sau khi hạn chế overfitting", fontsize=16, y=1.02)
fig.tight_layout()
plt.show()