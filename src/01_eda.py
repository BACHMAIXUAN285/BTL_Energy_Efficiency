# %% Import thư viện
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Tạo các thư mục cần thiết nếu chưa tồn tại
os.makedirs('./data/raw', exist_ok=True)
os.makedirs('./reports', exist_ok=True)

# %% Đọc dữ liệu gốc
# Thay đổi đường dẫn này trỏ tới file ENB2012_data.csv của bạn
data_path = 'data/raw/ENB2012_data.csv'
df = pd.read_csv(data_path)

# Đổi tên cột cho rõ nghĩa
column_names = {
    'X1': 'Relative_Compactness', 'X2': 'Surface_Area', 'X3': 'Wall_Area', 
    'X4': 'Roof_Area', 'X5': 'Overall_Height', 'X6': 'Orientation', 
    'X7': 'Glazing_Area', 'X8': 'Glazing_Area_Distribution', 
    'Y1': 'Heating_Load', 'Y2': 'Cooling_Load'
}
df = df.rename(columns=column_names)

# %% In thống kê mô tả
print("--- TÓM TẮT DỮ LIỆU ---")
print(df.info())
print("\n--- THỐNG KÊ CƠ BẢN ---")
print(df.describe())

# %% Vẽ biểu đồ phân phối của 2 biến mục tiêu (Heating Load & Cooling Load)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Heating_Load'], kde=True, color='red')
plt.title('Phân phối tải sưởi ấm (Heating Load)')
plt.xlabel('Heating Load')

plt.subplot(1, 2, 2)
sns.histplot(df['Cooling_Load'], kde=True, color='blue')
plt.title('Phân phối tải làm mát (Cooling Load)')
plt.xlabel('Cooling Load')

plt.tight_layout()
plt.savefig('reports/target_distribution.png')
print("\nĐã lưu biểu đồ phân phối tại: reports/target_distribution.png")

# %% Vẽ Ma trận tương quan (Correlation Heatmap)
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Ma trận Tương quan Pearson (Correlation Heatmap)')
plt.tight_layout()
plt.savefig('reports/correlation_heatmap.png')
print("Đã lưu ma trận tương quan tại: reports/correlation_heatmap.png")

# %% (Tùy chọn) Tìm ngoại lai bằng boxplot cho một vài biến tiêu biểu
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area']])
plt.title('Kiểm tra ngoại lai (Outliers) bằng Boxplot')
plt.savefig('reports/outliers_boxplot.png')
print("Đã lưu biểu đồ Boxplot tại: reports/outliers_boxplot.png")