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
# Initialize weights and bias
w1, w2, b = 0, 0, 0
# Perform gradient descent
