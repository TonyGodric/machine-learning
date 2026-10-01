"""3.27 Cho: w=[1, 2, -10]T, x=[3, 4, 1]T và điểm dữ liệu đã thêm bias.
Sử dụng phương pháp Perceptron:
1.	Tính wTx. 
2.	Xác định nhãn dự đoán của điểm dữ liệu. 
3.	Nếu nhãn thực tế là y=−1, điểm dữ liệu có bị phân lớp sai không?
"""
import numpy as np

w = np.array([[1], [2], [-10]])# vector trong so 
x = np.array([[3], [4], [1]]) # vector diem du lieu da them bias

wTx = np.dot(w.T, x) # Tính wTx
predicted_label = np.sign(wTx)[0][0] # 
actual_label = -1
is_misclassified = (predicted_label != actual_label)

print(f"w^T * x = {wTx[0][0]}")
print(f"Predicted label = {predicted_label}")
print(f"Is the data point misclassified for y = -1? {'Yes' if is_misclassified else 'No'}")

    