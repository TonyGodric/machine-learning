import pandas as pd
import string 
# Load the csv file 
data = pd.read_csv('data_gia_nha_hanoi.csv')

X= data[['bedrooms','area']] # dataset
Y= data['price'] # label

def predict_price(bedrooms,area,w1,w2,b):
    predicted_price = (w1 * bedrooms) + (w2 * area) + b
    return predicted_price

def mean_squared_error(bedrooms, area, w1, w2,b):
    n = len(bedrooms)
    total_error = 0
    for i in range(n):
        predicted_price = predict_price(bedrooms[i], area[i], w1, w2,b)
        total_error += (predicted_price - Y[i]) ** 2
    mse = total_error / n
    return mse
# gradient descent function
def gradient_descent(bedrooms, area, w1, w2,b, learning_rate, iterations):
    n = len(bedrooms)
    for i in range(iterations):
        total_error_w1 = 0
        total_error_w2 = 0
        total_error_b = 0
        for j in range(n):
            predicted_price = predict_price(bedrooms[j], area[j], w1, w2,b)
            error = predicted_price - Y[j]
            total_error_w1 += error * bedrooms[j]
            total_error_w2 += error * area[j]
            total_error_b += error
        w1 -= (learning_rate * total_error_w1) / n
        w2 -= (learning_rate * total_error_w2) / n
        b -= (learning_rate * total_error_b) / n
    return w1, w2,b
# Khởi tạo trọng số ban đầu, learning rate và số vòng lặp
w1 = 0.0
w2 = 0.0
b = 0.0
learning_rate = 0.0001  
iterations = 20000
bedrooms_list = X['bedrooms'].tolist()
area_list = X['area'].tolist()
price_list = Y.tolist()

# 2. Huấn luyện mô hình tìm w1, w2, b
w1, w2, b = gradient_descent(bedrooms_list, area_list, w1, w2, b, learning_rate, iterations)

print(f"Trọng số tối ưu: w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")

# 3. Đưa ra dự đoán cho một ngôi nhà mới (Ví dụ: 3 phòng ngủ, diện tích 70m2)
bedrooms_input = 3
area_input = 100

predicted = predict_price(bedrooms_input, area_input, w1, w2, b)
print(f"Dự đoán giá nhà ({bedrooms_input} phòng ngủ, {area_input}m2): {predicted:.2f} tỷ")