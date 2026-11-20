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
│   └── 03_modeling.py         # Script Huấn luyện, tinh chỉnh tham số và đánh giá mô hình (RMSE, R2)
│
├── .gitattributes             # Cấu hình Git LFS để quản lý các tệp dữ liệu/mô hình lớn
├── .gitignore                 # Cấu hình bỏ qua môi trường ảo (venv) và file rác khi đẩy lên Git
├── README.md                  # Tài liệu hướng dẫn dự án (Tệp này)
└── requirements.txt           # Danh sách các thư viện Python phụ thuộc của dự án
```

## ⚙️ Hướng dẫn cài đặt môi trường

Để đảm bảo code chạy ổn định và không xung đột thư viện, toàn bộ nhóm cần sử dụng môi trường ảo (Virtual Environment). Thực hiện lần lượt các lệnh sau trên Terminal (đối với Linux/WSL/macOS):

**1. Khởi tạo môi trường ảo:**
```bash
python3 -m venv venv
```

**2. Kích hoạt môi trường ảo:**
```bash
source venv/bin/activate
```
*(Trên Windows Command Prompt dùng lệnh: `venv\Scripts\activate`)*

**3. Cài đặt các thư viện yêu cầu:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Quy trình thực thi

Các thành viên chạy các kịch bản (script) theo đúng thứ tự dưới đây để thực hiện từ khâu lấy dữ liệu đến khi chạy ứng dụng web. Đảm bảo môi trường ảo `(venv)` đang được kích hoạt.

### Bước 1: Thu thập và Khám phá dữ liệu
*   **Tải dữ liệu thô:**
    ```bash
    python src/00_fetch_data.py
    ```
    *(Tải tập dữ liệu ENB2012 từ hệ thống UCI và lưu vào `data/raw/`)*
*   **Chạy Phân tích Khám phá (EDA):**
    ```bash
    python src/01_eda.py
    ```
    *(Phân tích thống kê, xuất các biểu đồ trực quan ra thư mục `reports/`)*

### Bước 2: Tiền xử lý dữ liệu
```bash
python src/02_preprocessing.py
```
*(Thực hiện làm sạch, mã hóa phân loại, chuẩn hóa Min-Max và lưu kết quả vào `data/processed/cleaned_energy_data.csv`)*

### Bước 3: Huấn luyện mô hình
```bash
python src/03_modeling.py
```
*(Đọc dữ liệu sạch, chia tập Train/Test, huấn luyện các mô hình Hồi quy, in ra chỉ số đánh giá và xuất mô hình tốt nhất vào `app/best_model.pkl`)*

### Bước 4: Chạy Ứng dụng Giao diện (Streamlit UI)
```bash
streamlit run app/main.py
```
*(Khởi chạy ứng dụng Web tại `http://localhost:8501`, cho phép nhập các thông số kiến trúc để dự báo tải năng lượng theo thời gian thực)*

## 🛠 Công nghệ sử dụng
*   **Ngôn ngữ:** Python 3.12+
*   **Thư viện Data & ML:** Pandas, NumPy, Scikit-Learn
*   **Trực quan hóa:** Matplotlib, Seaborn
*   **Giao diện Web UI:** Streamlit
*   **Quản lý phiên bản:** Git, Git LFS