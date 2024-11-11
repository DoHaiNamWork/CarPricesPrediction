import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoderpip

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv(r"main_trainning.csv")

print(data.head())

# Tách các đặc trưng và nhãn
X = data.drop('Gia', axis=1)  # Giả sử cột cuối cùng là cột nhãn
y = data['Gia']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train, y_train)

# Dự đoán nhãn cho dữ liệu kiểm tra
y_pred = knn.predict(X_test)

# Đánh giá hiệu suất của mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(knn, 'KNN.pkl')