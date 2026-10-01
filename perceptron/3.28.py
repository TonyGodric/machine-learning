"""3.28 Cho: w=[−2, 1, 0]T, x=[2, 3, 1]T,  y=1, trong đó, x đã thêm bias.
Sử dụng phương pháp Perceptron:
1.	Kiểm tra mẫu có bị phân lớp sai hay không. 
2.	Nếu sai, thực hiện một bước cập nhật Perceptron. 
3.	Tính lại giá trị wTx sau cập nhật."""
import numpy as np

# Khởi tạo
w = np.array([-2, 1, 0])  # Dùng mảng 1D cho gọn
x = np.array([2, 3, 1])
y = 1

# Bước 1: Tính wTx và dự đoán
wTx = np.dot(w, x)
predicted_label = np.sign(wTx)

print(f"w^T * x = {wTx}")
print(f"Predicted label = {predicted_label}")

# Bước 2: Kiểm tra và cập nhật
if predicted_label != y:
    print(f"Mẫu bị phân lớp sai (y={y}, dự đoán={predicted_label})")
    learning_rate = 1.0
    # Cập nhật theo công thức (y - y_hat)
    update = learning_rate * (y - predicted_label)
    w = w + update * x
    print(f"Trọng số sau cập nhật: {w}")
    
    # Bước 3: Tính lại wTx
    wTx_new = np.dot(w, x)
    print(f"w^T * x sau cập nhật = {wTx_new}")
else:
    print("Mẫu được phân lớp đúng, không cần cập nhật.")