# BTL Energy Efficiency

Dự án phân tích dữ liệu và xây dựng mô hình dự đoán hiệu quả năng lượng.

## Cấu trúc

- `data/raw/`: dữ liệu gốc, ví dụ `ENB2012_data.xlsx`.
- `data/processed/`: dữ liệu đã xử lý, ví dụ tệp CSV.
- `reports/`: biểu đồ và báo cáo được xuất từ quá trình phân tích.
- `src/01_eda.py`: phân tích khám phá dữ liệu.
- `src/02_preprocessing.py`: làm sạch và chuẩn bị dữ liệu.
- `src/03_modeling.py`: huấn luyện và đánh giá mô hình.
- `app/main.py`: giao diện Streamlit.

## Cài đặt

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo, sau đó cài các thư viện:

```bash
python -m pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app/main.py
```

Các script trong `src/` hiện là khung khởi đầu; cần triển khai các bước EDA,
tiền xử lý và mô hình hóa trước khi chạy quy trình dữ liệu hoàn chỉnh.
