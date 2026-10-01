from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
# 1. Tạo dữ liệu
X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=2.0, random_state=1
)
# 2. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)
# 3. Chuẩn hóa (quan trọng với Perceptron)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# 4. Khởi tạo và huấn luyện
model = Perceptron(
    eta0=0.1,
    max_iter=50,
    random_state=1,
    tol=None,
    shuffle=True,
    fit_intercept=True,
)
model.fit(X_train, y_train)
# 5. Dự đoán
y_pred = model.predict(X_test)
# 6. Đánh giá
print("Số epoch đã chạy:", model.n_iter_)
print("Accuracy:", accuracy_score(y_test, y_pred))