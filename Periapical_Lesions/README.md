# Medical Image Classification

## Mô tả
Dự án này thực hiện phân loại ảnh y tế sử dụng các thuật toán SVM và KNN. Các ảnh y tế được tổ chức theo từng lớp và lưu trong thư mục `data`.

## Cấu trúc
- `data/`: Chứa dữ liệu ảnh (các thư mục `train_images` và `test_images` được tổ chức theo nhãn).
- `src/`: Chứa mã nguồn của dự án.
- `requirements.txt`: Liệt kê các thư viện cần thiết.

## Hướng dẫn cài đặt
1. Cài đặt các thư viện bằng lệnh:
    ```bash
    pip install -r requirements.txt
    ```

2. Chạy tệp chính để huấn luyện và đánh giá mô hình:
    ```bash
    python src/main.py
    ```

## Yêu cầu
- Python 3.7 trở lên
- Các thư viện liệt kê trong `requirements.txt`
