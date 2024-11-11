import tkinter as tk
import joblib
import pandas as pd
from sklearn.ensemble._forest import ForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Đọc dữ liệu từ tập tin CSV
data = pd.read_csv(r"main_trainning.csv")

# Phân tách dữ liệu thành đặc trưng (features) và nhãn (label)
X = data.iloc[:, :-1]  # Các cột đặc trưng (tất cả trừ cột cuối cùng)
y = data.iloc[:, -1]   # Cột nhãn (cột cuối cùng)

# Khởi tạo Random Forest Regressor
model = RandomForestRegressor()

# Huấn luyện mô hình với tập huấn luyện
model.fit(X.values, y.values)

def submit_form():
    # Lấy dữ liệu từ các ô nhập
    data = []
    for entry in entry_fields:
        data.append(float(entry.get()))

    # Dự đoán trên tập dữ liệu
    y_pred = model.predict([data])
    ketqua = "Giá xe:"+ str(round(y_pred[0],0)) + " triệu đồng"
    result_label.config(text=ketqua, font=("Arial", 14), fg="red")

def reset_form():
    # Xóa nội dung đã nhập trong các ô nhập
    for entry in entry_fields:
        entry.delete(0, tk.END)
    # Reset nhãn kết quả
    result_label.config(text="Kết quả dự đoán:", font=("Arial", 14))


# Tạo cửa sổ
root = tk.Tk()
root.title("Dự đoán giá xe")

# Tạo Frame cho phần tiêu đề
title_frame = tk.Frame(root)
title_frame.grid(row=0, column=0, columnspan=2, pady=10)

# Thêm tiêu đề cho form
form_title = tk.Label(title_frame, text="Dự đoán giá xe", font=("Arial", 14, "bold"), fg="red")
form_title.pack()

# Tạo Frame cho phần nhập liệu
input_frame = tk.Frame(root)
input_frame.grid(row=1, column=0, columnspan=2)

# Tạo danh sách các tên của từng ô nhập
entry_names = ["Xuất xứ", "Tình trạng", "Dòng xe", "Số km đã đi", "Màu ngoại thất", "Màu nội thất",
               "Số cửa", "Số chỗ ngồi", "Hộp số", "Dẫn động", "Hãng", "Grade",
               "Năm sản xuất", "Loại động cơ", "Dung tích"]

# Tạo danh sách các ô nhập
entry_fields = []
for i in range(15):
    label = tk.Label(input_frame, text=f"{entry_names[i]}:", font=("Arial", 14), fg = "blue") # Chỉnh cỡ chữ ở đây
    label.grid(row=i, column=0, padx=10, pady=5, sticky="e") # Sử dụng sticky để căn chỉnh về phía đông
    entry = tk.Entry(input_frame, font=("Arial", 14)) # Chỉnh cỡ chữ ở đây
    entry.grid(row=i, column=1, padx=10, pady=5, sticky="w") # Sử dụng sticky để căn chỉnh về phía tây
    entry_fields.append(entry)

# Tạo nút "Submit" để gửi dữ liệu
submit_button = tk.Button(root, text="Submit", command=submit_form, font=("Arial", 12)) # Chỉnh cỡ chữ ở đây
submit_button.grid(row=2, column=0, padx=10, pady=10)

# Tạo nút "Reset" để xóa dữ liệu đã nhập
reset_button = tk.Button(root, text="Reset", command=reset_form, font=("Arial", 12)) # Chỉnh cỡ chữ ở đây
reset_button.grid(row=2, column=1, padx=10, pady=10)

# Tạo Frame cho phần kết quả
result_frame = tk.Frame(root)
result_frame.grid(row=3, column=0, columnspan=2)

result_label = tk.Label(result_frame, text="Giá xe:", font=("Arial", 14)) # Chỉnh cỡ chữ ở đây
result_label.pack()
root.geometry("500x800") # Thay đổi kích thước cửa sổ

root.mainloop()
