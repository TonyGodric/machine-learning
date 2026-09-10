# Du bao gia nha hanoi bang linear regression
## Bai toan 
    Su dung Linear Regression de du doan gia nha dua tren:
        bedrooms: so phong ngu
        area: dien tich nha (m2)
        price: gia nha (ty dong)
    Mo hinh su dung hai dac trung dau vao la bedrooms va area de du doan price.
## Du lieu
    File: machine learning/data_gia_nha_hanoi.csv
    Cot: bedrooms, area , price
    Dong: 8
    Dataset:
    bedrooms,area,price
    2,45,3.8
    3,37,4.25
    4,40,2.6
    4,52,4.8 
    5,90,9.7
    3,32,1.95
    2,68,3.74
    2,58,2.43
## Cach chay
    Python :machine learning/exeday2.py
    Su dung thu vien pandas de doc file CSV
## Phuong phap 
    1 Hoc mo hinh :
        predicted_price = bedrooms * w1 + area * w2 + b
        Trong do:
            bedrooms, area: cac feature dau vao.
            w1, w2: trong so (weight), the hien muc do anh huong cua tung feature den ket qua du doan.
            b: bias, la gia tri dieu chinh co dinh cho mo hinh.
            predicted_price: gia nha ma mo hinh du doan.
            Muc tieu la tim duoc w1, w2 va b sao cho gia du doan gan voi gia thuc te nhat.
    2 Su dung ham mat mat(loss function):
        Binh phuong toi thieu (mean squared error):
        MSE = (1/n)* Σ (price - predicted_price)^2
        Giai thich :MSEla muc do do su sai lech giua mo hinh du doan va gia thuc te tuc no la thuoc do xem mo hinh sai bao nhieu. Target la tim w va b de MSE nho nhat.
    3 Su dung gradient descent de toi uu:
    De tim w1, w2 va b, chuong trinh su dung Gradient Descent.
    Ban dau:
    w1 = 0
    w2 = 0
    b  = 0
    Sau do thuc hien nhieu vong lap.
    Moi vong lap gom cac buoc:
    Tinh gia nha du doan cho tung du lieu.
    Tinh sai so:
    error = predicted_price - actual_price
    Tinh gradient cho w1, w2 va b.
    Cap nhat cac tham so theo huong lam giam sai so:
    w1 = w1 - learning_rate * gradient_w1
    w2 = w2 - learning_rate * gradient_w2
    b  = b  - learning_rate * gradient_b
    Gradient Descent duoc lap lai nhieu lan de cac tham so dan tien den gia tri phu hop. 
## Ket qua
    Trong so sau khi train ( learning_rate=0.0001):
        w1 = 0
        w2 = 0
        b  = 0
    Iterations	w1	w2	b	Dự đoán (3PN, 100m2)
    1000	0.0707	0.0774	-0.0115	7.94 tỷ
    5000	0.2599	0.0680	-0.0732	7.51 tỷ
    20000	0.5011	0.0596	-0.3566	7.10 tỷ
    Sau 20.000 iterations, mo hinh thu duoc gan:
    w1 = 0.5011
    w2 = 0.0596
    b  = -0.3566
    Mo hinh luc nay co dang:
    price ≈ 0.5011 * bedrooms + 0.0596 * area - 0.3566
    Voi mot ngoi nha co:
    bedrooms = 3
    area = 100m2
    mo hinh du doan xap xi 7.1 ty 
## Nhan xet:
    Qua ket qua tren, co the thay w1 va w2 thay doi voi toc do khac nhau trong qua trinh huan luyen.

    Nguyen nhan chinh la hai feature co thang do khac nhau:

    bedrooms chi co gia tri khoang 2-5.
    area co gia tri khoang 32-90.

    Do chua chuan hoa du lieu, gradient cua hai trong so co do lon khac nhau. Vi vay Gradient Descent khong cap nhat hai trong so voi toc do giong nhau.

    Sau 20.000 iterations, gia du doan van thay doi so voi cac moc truoc do, nen chua the ket luan mo hinh da hoi tu hoan toan.
## Han che
    Dataset hien tai chi co 8 dong, vi vay mo hinh chua du du lieu de dua ra du doan dang tin cay cho gia nha thuc te.

    Ngoai ra:

        Chua chuan hoa cac feature.
        So luong du lieu qua it.
        Gia nha thuc te con phu thuoc vao nhieu yeu to khac nhu vi tri, quan, mat duong, noi that, huong nha,...
        Chua co tap du lieu rieng de kiem tra (test set).
        Chua danh gia mo hinh bang cac chi so khac ngoai qua trinh toi uu MSE.

    Do do, ket qua 7.10 ty chi mang tinh minh hoa cho qua trinh xay dung va huan luyen Linear Regression, khong nen xem la gia nha thuc te.

## Huong phat trien

    Co the cai thien mo hinh bang cach:

    Thu thap them du lieu.
    Chuan hoa cac feature truoc khi huan luyen.
    Chia du lieu thanh tap train va test.
    Thu nghiem learning rate va so iterations khac nhau.
    Bo sung them cac feature nhu vi tri, so tang, khoang cach den trung tam,...
    So sanh voi cac phuong phap hoi quy khac.

