"""3.27 Cho: w=[1, 2, -10]T, x=[3, 4, 1]T và điểm dữ liệu đã thêm bias.
Sử dụng phương pháp Perceptron:
1.	Tính wTx. 
2.	Xác định nhãn dự đoán của điểm dữ liệu. 
3.	Nếu nhãn thực tế là y=−1, điểm dữ liệu có bị phân lớp sai không?
"""
import numpy as np
import matplotlib.pyplot as plt

w = np.array([[1], [2], [-10]])
x = np.array([[3], [4], [1]])

wTx = np.dot(w.T, x)
predicted_label = np.sign(wTx)[0][0]
actual_label = -1
is_misclassified = (predicted_label != actual_label)

print(f"w^T * x = {wTx[0][0]}")
print(f"Predicted label = {predicted_label}")
print(f"Is the data point misclassified for y = -1? {'Yes' if is_misclassified else 'No'}")
def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0:
        x11, x12 = -100, 100
        return plt.plot([x11, x12], [-(w1*x11 + w0)/w2, -(w1*x12 + w0)/w2], 'k')
    else:
        x10 = -w0/w1
        return plt.plot([x10, x10], [-100, 100], 'k')
    