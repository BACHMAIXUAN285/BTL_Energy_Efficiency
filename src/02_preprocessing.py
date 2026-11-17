# %% Import thư viện
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Tạo thư mục lưu dữ liệu đã xử lý
os.makedirs('./data/processed', exist_ok=True)

# %% Đọc dữ liệu
data_path = 'data/raw/ENB2012_data.csv'
df = pd.read_csv(data_path)

column_names = {
    'X1': 'Relative_Compactness', 'X2': 'Surface_Area', 'X3': 'Wall_Area', 
    'X4': 'Roof_Area', 'X5': 'Overall_Height', 'X6': 'Orientation', 
    'X7': 'Glazing_Area', 'X8': 'Glazing_Area_Distribution', 
    'Y1': 'Heating_Load', 'Y2': 'Cooling_Load'
}
df = df.rename(columns=column_names)

# In ra số chiều dữ liệu ban đầu
print(f"Kích thước dữ liệu gốc: {df.shape}")

# %% BƯỚC 1: Xử lý Missing Values và Outliers
# Kiểm tra missing values (Dataset này thực tế không có missing data)
print(f"Số lượng missing values:\n{df.isnull().sum()}")

# LƯU Ý VỀ OUTLIERS: Dữ liệu bài toán này là kết quả mô phỏng nhiệt động lực học tòa nhà.
# Việc áp dụng xóa outlier bằng IQR có thể làm mất đi các hình thái kiến trúc hợp lệ ở các điểm biên.
# Do đó, trong bước này chúng ta GIỮ NGUYÊN không xóa dòng nào để đảm bảo tính toàn vẹn vật lý.

# %% BƯỚC 2: Mã hóa phân loại (One-Hot Encoding)
# Biến X6 (Orientation) và X8 (Glazing Area Distribution) mang tính phân loại, không phải định lượng số học.
categorical_cols = ['Orientation', 'Glazing_Area_Distribution']

# Sử dụng pd.get_dummies với drop_first=True để tránh bẫy đa cộng tuyến (dummy variable trap)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"\nKích thước dữ liệu sau One-Hot Encoding: {df_encoded.shape}")

# %% BƯỚC 3: Chuẩn hóa dải giá trị (Min-Max Normalization)
# Tách riêng features (đầu vào) và targets (mục tiêu)
targets = ['Heating_Load', 'Cooling_Load']
features = [col for col in df_encoded.columns if col not in targets]

X = df_encoded[features]
Y = df_encoded[targets]

# Khởi tạo MinMaxScaler
scaler = MinMaxScaler()

# fit_transform lên tập X và chuyển lại thành DataFrame
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# In thử 5 dòng đầu sau chuẩn hóa
print("\nDữ liệu Features sau khi chuẩn hóa Min-Max (5 dòng đầu):")
print(X_scaled.head())

# %% BƯỚC 4: Ghép nối và Xuất dữ liệu sạch
# Ghép X (đã scale) và Y (chưa scale, giữ nguyên giá trị để dự báo dễ đọc)
df_processed = pd.concat([X_scaled, Y], axis=1)

# Xuất ra file CSV
output_path = 'data/processed/cleaned_energy_data.csv'
df_processed.to_csv(output_path, index=False)

print(f"\n[HOÀN TẤT] Dữ liệu đã được tiền xử lý và lưu tại: {output_path}")