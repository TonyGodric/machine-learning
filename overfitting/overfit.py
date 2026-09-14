import pandas as pd
import numpy as np
datatrain = pd.read_csv('data train.csv')
datatest = pd.read_csv('data test.csv')
datavalidation = pd.read_csv('data validation.csv')
X = datatrain['area'] #dataset
Y = datatrain['price'] #label

def predict_price(area, w, b):
    predicted_price = (w * area) + b
    return predicted_price
def mean_squared_error(area, w, b):
    n = len(area)
    total_error = 0
    for i in range(n):
        predicted_price = predict_price(area[i], w, b)
        total_error += (predicted_price - Y[i]) ** 2
    mse = total_error / n
    return mse
def gradient_descent(area, w, b, learning_rate, iterations):
    n = len(area)
    for i in range(iterations):
        total_error_w = 0
        total_error_b = 0
        for j in range(n):
            predicted_price = predict_price(area[j], w, b)
            error = predicted_price - Y[j]
            total_error_w += error * area[j]
            total_error_b += error
        w -= (learning_rate * total_error_w) / n
        b -= (learning_rate * total_error_b) / n
    return w, b
# gio thi lam the nao voi data train de overfit nhung khi test thi lai khong tot
def overfit_model(area, w, b, learning_rate, iterations):
    n = len(area)
    for i in range(iterations):
        total_error_w = 0
        total_error_b = 0
        for j in range(n):
            predicted_price = predict_price(area[j], w, b)
            error = predicted_price - Y[j]
            total_error_w += error * area[j]
            total_error_b += error
        w -= (learning_rate * total_error_w) / n
        b -= (learning_rate * total_error_b) / n
    return w, b
def train_model(area, w, b, learning_rate, iterations):
    n = len(area)
    for i in range(iterations):
        total_error_w = 0
        total_error_b = 0
        for j in range(n):
            predicted_price = predict_price(area[j], w, b)
            error = predicted_price - Y[j]
            total_error_w += error * area[j]
            total_error_b += error
        w -= (learning_rate * total_error_w) / n
        b -= (learning_rate * total_error_b) / n
    return w, b
def test_model(area, w, b):
    n = len(area)
    total_error = 0
    for i in range(n):
        predicted_price = predict_price(area[i], w, b)
        total_error += (predicted_price - Y[i]) ** 2
    mse = total_error / n
    return mse
def main():
    # Khởi tạo trọng số ban đầu, learning rate và số vòng lặp
    w = 0.0
    b = 0.0
    learning_rate = 0.0001  
    iterations = 20000

    area_list = X.tolist()
    price_list = Y.tolist()

    # Huấn luyện mô hình tìm w, b
    w, b = train_model(area_list, w, b, learning_rate, iterations)

    print(f"Trọng số tối ưu: w = {w:.4f}, b = {b:.4f}")

    # Đưa ra dự đoán cho một ngôi nhà mới (Ví dụ: diện tích 100m2)
    area_input = 100

    predicted = predict_price(area_input, w, b)
    print(f"Dự đoán giá nhà ({area_input}m2): {predicted:.2f} tỷ")
    # gio thi dung validation set de kiem tra xem model co bi overfit hay khong
    def validate_model(area, w, b):
        n = len(area)
        total_error = 0
        for i in range(n):
            predicted_price = predict_price(area[i], w, b)
            total_error += (predicted_price - Y[i]) ** 2
        mse = total_error / n
        return mse
    
