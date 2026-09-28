"""3.29 Viết code để xây dựng lớp Perceptron có hàm fit để huấn luyện mô hình và hàm predict để dự báo nhãn của dữ liệu mới."""
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Convert labels to -1 and 1
        y = np.where(y <= 0, -1, 1)

        for _ in range(self.max_iterations):
            for i in range(X.shape[0]):
                # Compute prediction
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = np.sign(linear_output)

                # Update weights and bias if misclassified
                if prediction != y[i]:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

def draw_line(weights, bias):
    if weights[1] != 0:
        x_vals = np.array([-10, 10])
        y_vals = -(weights[0] * x_vals + bias) / weights[1]
        plt.plot(x_vals, y_vals, 'k-')
    else:
        x_val = -bias / weights[0]
        plt.axvline(x=x_val, color='k')
def main():
    # Sample data
    X = np.array([[2, 2], [4, 2], [4, 4], [2, 4]])
    y = np.array([1, 1, -1, -1])  # Labels

    # Create and train Perceptron
    perceptron = Perceptron(learning_rate=0.1, max_iterations=10)
    perceptron.fit(X, y)

    # Predict on training data
    predictions = perceptron.predict(X)
    print("Predictions:", predictions)

    # Plotting
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    draw_line(perceptron.weights, perceptron.bias)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.title("Perceptron Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid()
    plt.show()
if __name__ == "__main__":
    main()
    