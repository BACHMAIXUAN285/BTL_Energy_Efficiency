# BTL Khai Phá Dữ Liệu - Dự Đoán Năng Lượng Tòa Nhà (Energy Efficiency)

Dự án áp dụng toàn diện quy trình khám phá tri thức (KDD / CRISP-DM) để dự báo nhu cầu năng lượng sưởi ấm (Heating Load) và làm mát (Cooling Load) của tòa nhà dựa trên các đặc trưng kiến trúc.

## Cấu trúc dự án

Dự án được phân chia thành các module độc lập, tách biệt rõ ràng giữa dữ liệu, mã nguồn xử lý, báo cáo và giao diện người dùng:

```text
BTL_Energy_Efficiency/
├── app/
│   ├── main.py                # Mã nguồn chạy giao diện ứng dụng trực quan (Streamlit)
│   └── best_model.pkl         # Tệp mô hình tốt nhất đã được huấn luyện (sinh ra từ bước 3)
│
├── data/
│   ├── processed/             # Dữ liệu sạch, đã chuẩn hóa và sẵn sàng cho mô hình hóa (.csv)
│   └── raw/                   # Dữ liệu gốc chưa qua xử lý tải về từ UCI
│
├── reports/                   # Thư mục lưu trữ các biểu đồ EDA (phân phối, ma trận tương quan)
│
├── src/
│   ├── 00_fetch_data.py       # Script kết nối API lấy dữ liệu tự động từ thư viện ucimlrepo
│   ├── 01_eda.py              # Script Khám phá dữ liệu (EDA): Thống kê mô tả và trực quan hóa
│   ├── 02_preprocessing.py    # Script Tiền xử lý: Mã hóa One-Hot, chuẩn hóa Min-Max
│   ├── 03_modeling.py         # Script mô hình hóa (bản hiện có)
│   └── 04_evaluation.py       # Đánh giá chéo mô hình, chống rò rỉ giữa các fold
│
├── .gitattributes             # Cấu hình Git LFS để quản lý các tệp dữ liệu/mô hình lớn
├── .gitignore                 # Cấu hình bỏ qua môi trường ảo (venv) và file rác khi đẩy lên Git
├── README.md                  # Tài liệu hướng dẫn dự án (Tệp này)
└── requirements.txt           # Danh sách các thư viện Python phụ thuộc của dự án
```

## ⚙️ Hướng dẫn cài đặt môi trường

Để tránh xung đột thư viện, nên sử dụng môi trường ảo và chạy các lệnh sau trong PowerShell trên Windows.

Mở PowerShell tại thư mục gốc `BTL_Energy_Efficiency`.

**1. Tạo môi trường ảo (bỏ qua nếu đã có `venv`):**
```powershell
py -3.12 -m venv venv
```

**2. Cài đặt các thư viện yêu cầu:**
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Các lệnh bên dưới gọi trực tiếp Python trong môi trường ảo nên không cần activate.

## Quy trình thực thi

Các lệnh PowerShell dưới đây được chạy từ thư mục gốc dự án. Bước 3 có thể chạy độc lập, không cần chạy trước bước tiền xử lý.

### Bước 1: Thu thập và Khám phá dữ liệu

Nếu chưa có `data/raw/ENB2012_data.csv`, tải dữ liệu từ UCI:
```powershell
.\venv\Scripts\python.exe src\00_fetch_data.py
```

Chạy EDA để tạo các biểu đồ trong `reports/`:
```powershell
.\venv\Scripts\python.exe src\01_eda.py
```

### Bước 2: Tiền xử lý dữ liệu
```powershell
.\venv\Scripts\python.exe src\02_preprocessing.py
```

Bước này tạo CSV đã xử lý cho các mục đích khác. Step 3 không đọc tệp đó vì tiền xử lý cần được fit riêng trong mỗi fold.

### Bước 3: Đánh giá mô hình dự báo tải năng lượng

`src/04_evaluation.py` đọc trực tiếp dữ liệu gốc CSV hoặc Excel, sau đó fit tiền xử lý riêng trên phần train của từng fold. Mặc định script dùng `data/raw/ENB2012_data.csv`; có thể truyền đường dẫn khác bằng `--data`.

Chạy baseline Linear Regression:
```powershell
.\venv\Scripts\python.exe src\04_evaluation.py --baseline-only
```

Chạy đủ sáu mô hình:
```powershell
.\venv\Scripts\python.exe src\04_evaluation.py
```

Để chỉ định CSV hoặc Excel khác, thêm `--data`, ví dụ:
```powershell
.\venv\Scripts\python.exe src\04_evaluation.py --data data\raw\ENB2012_data.csv --baseline-only
```

Kết quả từng fold và bảng tổng hợp được lưu lần lượt tại `reports/evaluation/fold_metrics.csv` và `reports/evaluation/cv_summary.csv`. KFold chia ngẫu nhiên theo dòng; GroupKFold giữ các dòng có cùng tổ hợp hình học X1–X5 ở cùng fold validation, để đo khả năng dự báo cho cấu hình hình học chưa gặp khi huấn luyện.

### Bước 4: Ứng dụng giao diện (chưa triển khai)

`app/main.py` hiện chưa có nội dung ứng dụng Streamlit; bước này chưa chạy được trong phiên bản hiện tại.

## 🛠 Công nghệ sử dụng
*   **Ngôn ngữ:** Python 3.12+
*   **Thư viện Data & ML:** Pandas, NumPy, Scikit-Learn
*   **Trực quan hóa:** Matplotlib, Seaborn
*   **Giao diện Web UI:** Streamlit
*   **Quản lý phiên bản:** Git, Git LFS