# %% Import thư viện
import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

# Đảm bảo thư mục raw tồn tại
os.makedirs('./data/raw', exist_ok=True)

print("Đang kết nối tới UCI Machine Learning Repository...")
print("Đang tải dataset Energy Efficiency (ID=242)...")

# %% Lấy dữ liệu qua API của ucimlrepo
try:
    energy_efficiency = fetch_ucirepo(id=242)
    
    # Lấy features (X) và targets (y)
    X = energy_efficiency.data.features
    y = energy_efficiency.data.targets
    
    # Gộp toàn bộ lại thành một DataFrame duy nhất
    df_raw = pd.concat([X, y], axis=1)
    
    print("\nTrích xuất thành công! Xem trước 5 dòng đầu:")
    print(df_raw.head())
    
    # %% Lưu dữ liệu xuống thư mục cục bộ
    # Lưu dưới định dạng CSV thay vì Excel để tối ưu tốc độ đọc/ghi
    output_path = './data/raw/ENB2012_data.csv'
    df_raw.to_csv(output_path, index=False)
    
    print(f"\n[HOÀN TẤT] Dữ liệu thô đã được lưu tự động tại: {output_path}")

except Exception as e:
    print(f"Có lỗi xảy ra trong quá trình tải dữ liệu: {e}")