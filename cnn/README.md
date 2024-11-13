# Dự án nhận dạng Chó và Mèo

## Mô tả
Dự án này sử dụng mô hình CNN để phân loại ảnh giữa hai loại động vật: Chó và Mèo.

## Cài đặt

1. Cài đặt các thư viện Python cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

2. Đảm bảo dữ liệu của bạn được tổ chức như sau:
    ```
    data/
    ├── train/
    │   ├── dog/                # 80 ảnh chó
    │   └── cat/                # 80 ảnh mèo
    └── validation/
        ├── dog/                # 20 ảnh chó
        └── cat/                # 20 ảnh mèo
    ```

## Huấn luyện mô hình
Để huấn luyện mô hình, chạy tệp `train.py`:
```bash
python src/train.py
python src/predict.py <đường_dẫn_đến_ảnh>

---

### Cách sử dụng:

1. **Cài đặt môi trường**:
   - Tạo một môi trường ảo và cài đặt các thư viện yêu cầu:
     ```bash
     pip install -r requirements.txt
     ```

2. **Chuẩn bị dữ liệu**:
   - Đặt ảnh huấn luyện và kiểm tra vào các thư mục `data/train/dog/`, `data/train/cat/`, `data/validation/dog/`, `data/validation/cat/`.

3. **Huấn luyện mô hình**:
   - Chạy lệnh sau để huấn luyện mô hình:
     ```bash
     python src/train.py
     ```

4. **Dự đoán với mô hình**:
   - Sau khi huấn luyện, bạn có thể sử dụng mô hình đã huấn luyện để dự đoán ảnh mới:
     ```bash
     python src/predict.py <đường_dẫn_đến_ảnh>
     ```

Hy vọng mã nguồn trên sẽ giúp bạn hoàn thành bài tập nhận dạng động vật sử dụng CNN!
