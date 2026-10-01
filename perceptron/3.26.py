def f(x):
    return x**2 - 4*x + 5
def df(x):
    return 2*x - 4
# cho biet do doc cua ham tai x
def gradient_descent(x0, learning_rate, num_steps):
    x = x0 
    values = [f(x)]
    for _ in range(num_steps):
        x = x - learning_rate * df(x)
        values.append(f(x))
    return values
def main():
    x0 = 5 # gia tri ban dau cua x 
    learning_rate = 0.2 # kich thuoc moi buoc cap nhap
    num_steps = 4 # so buoc cap nhat
    values = gradient_descent(x0, learning_rate, num_steps) # lu gia tri cua ham so tai moi buoc cap nhat
    for i, value in enumerate(values):
        print(f"Step {i}: f(x) = {value}")
if __name__ == "__main__":
    main()
    print("Nhận xét: Thuật toán hội tụ nhanh chóng về giá trị tối thiểu của hàm số, với các bước cập nhật giảm dần giá trị hàm số.")