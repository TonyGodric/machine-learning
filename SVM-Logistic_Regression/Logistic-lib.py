import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

np.random.seed(2)

# Dữ liệu — dùng reshape(-1, 1) để luôn ra (20, 1)
X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]
            ).reshape(-1, 1)          # <-- (20, 1)

y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

print("X.shape =", X.shape)   # phải là (20, 1)
print("y.shape =", y.shape)   # phải là (20,)

# Huấn luyện
model = LogisticRegression()
model.fit(X, y)

w0 = model.intercept_[0]
w1 = model.coef_[0][0]
print("w0 =", w0, " w1 =", w1)

threshold = -w0 / w1
print("threshold =", threshold)

# ---- Vẽ ----
X0 = X[y == 0].ravel()
X1 = X[y == 1].ravel()

plt.plot(X0, np.zeros_like(X0), 'ro', markersize=8, label='Fail')
plt.plot(X1, np.ones_like(X1),  'bs', markersize=8, label='Pass')

xx = np.linspace(0, 6, 1000).reshape(-1, 1)   # cũng phải reshape
yy = model.predict_proba(xx)[:, 1]

plt.axis([-2, 8, -1, 2])
plt.plot(xx.ravel(), yy, 'g-', linewidth=2, label='P(pass)')
plt.plot(threshold, 0.5, 'y^', markersize=10,
         label=f'threshold={threshold:.2f}')
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.legend()
plt.show()