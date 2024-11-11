# Import thư viện cần thiết
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

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo Random Forest Regressor
model = RandomForestRegressor(random_state=42)

# Huấn luyện mô hình với tập huấn luyện
model.fit(X_train.values, y_train.values)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test.values)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

joblib.dump(model, 'Random_Forest.pkl')