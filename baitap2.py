from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
RANDOM_SEED = 42
SPLIT_SEED = 10
N_SAMPLES = 90

rng = np.random.default_rng(RANDOM_SEED)
x = np.sort(rng.uniform(-3, 3, N_SAMPLES))
y_true = np.sin(1.5 * x) + 0.25 * x
y = y_true + rng.normal(loc=0, scale=0.25, size=N_SAMPLES)

# Tạo ba tập dữ liệu theo chỉ số; test được giữ lại để đánh giá cuối cùng.
all_idx = np.arange(N_SAMPLES)
train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.40, random_state=SPLIT_SEED
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=SPLIT_SEED
)

split = np.empty(N_SAMPLES, dtype=object)
split[train_idx] = "train"
split[val_idx] = "validation"
split[test_idx] = "test"

df = pd.DataFrame({"x": x, "y": y, "y_true": y_true, "split": split})
data_path = Path("du_lieu_overfitting.csv")
df.to_csv(data_path, index=False)

print(f"Đã tạo: {data_path.resolve()}")
print(df["split"].value_counts())
df.head(10)
colors = {"train": "#2563EB", "validation": "#F59E0B", "test": "#DC2626"}

fig, ax = plt.subplots(figsize=(10, 5.5))
for group in ["train", "validation", "test"]:
    part = df[df["split"] == group]
    ax.scatter(part["x"], part["y"], s=48, alpha=0.78,
               label=group.capitalize(), color=colors[group])

ax.plot(x, y_true, color="black", linewidth=2.2, label="Quan hệ thật")
ax.set(title="Bộ dữ liệu phi tuyến có nhiễu", xlabel="x", ylabel="y")
ax.legend(ncol=4)
plt.show()
X = df[["x"]].to_numpy()
y = df["y"].to_numpy()

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]


def make_polynomial_model(degree, estimator):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def regression_metrics(model, X_part, y_part):
    pred = model.predict(X_part)
    return {
        "RMSE": mean_squared_error(y_part, pred) ** 0.5,
        "R2": r2_score(y_part, pred),
    }


overfit_model = make_polynomial_model(20, LinearRegression())
overfit_model.fit(X_train, y_train)

overfit_metrics = pd.DataFrame({
    "Train": regression_metrics(overfit_model, X_train, y_train),
    "Validation": regression_metrics(overfit_model, X_val, y_val),
    "Test": regression_metrics(overfit_model, X_test, y_test),
}).T

overfit_metrics
x_grid = np.linspace(-3, 3, 700).reshape(-1, 1)
y_grid_true = np.sin(1.5 * x_grid.ravel()) + 0.25 * x_grid.ravel()

fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(X_train[:, 0], y_train, s=52, alpha=0.82,
           color=colors["train"], label="Train")
ax.scatter(X_val[:, 0], y_val, s=52, alpha=0.82,
           color=colors["validation"], label="Validation")
ax.scatter(X_test[:, 0], y_test, s=52, alpha=0.82,
           color=colors["test"], label="Test")
ax.plot(x_grid, y_grid_true, "k--", linewidth=2.2, label="Quan hệ thật")
ax.plot(x_grid, overfit_model.predict(x_grid), color="#7C3AED", linewidth=2.5,
        label="Đa thức bậc 20")
ax.set_ylim(-4, 4)
ax.set(title="Mô hình overfit: đường dự đoán dao động mạnh để khớp nhiễu",
       xlabel="x", ylabel="y")
ax.legend(ncol=3)
ax.text(0.02, 0.03,
        "Một phần đường dự đoán vượt khỏi khung y = [-4, 4].",
        transform=ax.transAxes, fontsize=10)
plt.show()
degrees = range(1, 21)
train_rmse, val_rmse = [], []

for degree in degrees:
    model = make_polynomial_model(degree, LinearRegression())
    model.fit(X_train, y_train)
    train_rmse.append(mean_squared_error(y_train, model.predict(X_train)) ** 0.5)
    val_rmse.append(mean_squared_error(y_val, model.predict(X_val)) ** 0.5)

best_degree_val = int(np.argmin(val_rmse) + 1)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(degrees, train_rmse, marker="o", label="Train RMSE", color="#2563EB")
ax.plot(degrees, val_rmse, marker="o", label="Validation RMSE", color="#F59E0B")
ax.axvline(best_degree_val, color="#059669", linestyle="--",
           label=f"Bậc tốt nhất theo validation = {best_degree_val}")
ax.set_xticks(list(degrees))
ax.set(title="Đường cong độ phức tạp của mô hình",
       xlabel="Bậc đa thức", ylabel="RMSE (càng thấp càng tốt)")
ax.legend()
plt.show()

print(f"Bậc được chọn từ tập validation: {best_degree_val}")
train_val_idx = np.concatenate([train_idx, val_idx])
X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]

simple_model = make_polynomial_model(best_degree_val, LinearRegression())
simple_model.fit(X_train_val, y_train_val)

print("Mô hình giảm độ phức tạp")
print(f"- Bậc đa thức: {best_degree_val}")
print(f"- Test RMSE: {regression_metrics(simple_model, X_test, y_test)['RMSE']:.4f}")
print(f"- Test R2:   {regression_metrics(simple_model, X_test, y_test)['R2']:.4f}")
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

ridge_degree20 = make_polynomial_model(20, Ridge(max_iter=10000))
ridge_search = GridSearchCV(
    estimator=ridge_degree20,
    param_grid={"model__alpha": np.logspace(-6, 3, 25)},
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
)
ridge_search.fit(X_train_val, y_train_val)
ridge_model = ridge_search.best_estimator_

print("Ridge trên đa thức bậc 20")
print(f"- Alpha tốt nhất: {ridge_search.best_params_['model__alpha']:.6g}")
print(f"- CV RMSE:        {-ridge_search.best_score_:.4f}")
print(f"- Test RMSE:      {regression_metrics(ridge_model, X_test, y_test)['RMSE']:.4f}")
print(f"- Test R2:        {regression_metrics(ridge_model, X_test, y_test)['R2']:.4f}")
tuned_pipeline = make_polynomial_model(1, Ridge(max_iter=10000))

param_grid = {
    "poly__degree": list(range(1, 16)),
    "model__alpha": np.logspace(-4, 3, 15),
}

tuned_search = GridSearchCV(
    estimator=tuned_pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
)
tuned_search.fit(X_train_val, y_train_val)
tuned_model = tuned_search.best_estimator_

print("Mô hình được chọn bằng cross-validation")
print(f"- Degree tốt nhất: {tuned_search.best_params_['poly__degree']}")
print(f"- Alpha tốt nhất:  {tuned_search.best_params_['model__alpha']:.6g}")
print(f"- CV RMSE:         {-tuned_search.best_score_:.4f}")
print(f"- Test RMSE:       {regression_metrics(tuned_model, X_test, y_test)['RMSE']:.4f}")
print(f"- Test R2:         {regression_metrics(tuned_model, X_test, y_test)['R2']:.4f}")
models = {
    "Overfit: degree=20, không phạt": overfit_model,
    f"Giảm độ phức tạp: degree={best_degree_val}": simple_model,
    "Ridge: degree=20, alpha chọn bằng CV": ridge_model,
    "Tối ưu degree + alpha bằng CV": tuned_model,
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
for ax, (title, model) in zip(axes.ravel(), models.items()):
    ax.scatter(X_train_val[:, 0], y_train_val, s=30, alpha=0.55,
               color="#2563EB", label="Train + Validation")
    ax.scatter(X_test[:, 0], y_test, s=45, alpha=0.85,
               color="#DC2626", label="Test")
    ax.plot(x_grid, y_grid_true, "k--", linewidth=1.8, label="Quan hệ thật")
    ax.plot(x_grid, model.predict(x_grid), color="#7C3AED", linewidth=2.2,
            label="Dự đoán")
    ax.set_title(title)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
fig.suptitle("So sánh mô hình trước và sau khi chống overfitting", y=1.05, fontsize=16)
fig.tight_layout()
plt.show()
comparison_rows = []
for name, model in models.items():
    fit_X = X_train if model is overfit_model else X_train_val
    fit_y = y_train if model is overfit_model else y_train_val
    comparison_rows.append({
        "Mô hình": name,
        "RMSE trên dữ liệu dùng để fit": regression_metrics(model, fit_X, fit_y)["RMSE"],
        "Test RMSE": regression_metrics(model, X_test, y_test)["RMSE"],
        "Test R2": regression_metrics(model, X_test, y_test)["R2"],
    })

comparison = pd.DataFrame(comparison_rows).sort_values("Test RMSE")
comparison