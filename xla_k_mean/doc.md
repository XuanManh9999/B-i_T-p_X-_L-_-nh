# K-Means Clustering on Iris Dataset

Đây là một ứng dụng đơn giản của thuật toán phân cụm K-Means trên tập dữ liệu Iris, sử dụng các thư viện NumPy và Scikit-learn. Mã này cũng đánh giá chất lượng của mô hình phân cụm bằng nhiều chỉ số khác nhau.

## Nội dung

1. **Chuẩn bị dữ liệu**
2. **Khởi tạo các tham số cho thuật toán K-Means**
3. **Tính toán khoảng cách và cập nhật nhãn**
4. **Cập nhật vị trí của các centroid**
5. **Chạy thuật toán K-Means**
6. **Đánh giá kết quả**
7. **In kết quả**

## Chi tiết mã

### 1. Chuẩn bị dữ liệu IRIS

```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y_true = iris.target
```
- `load_iris()`: Tải tập dữ liệu Iris, bao gồm 150 mẫu với 4 đặc trưng cho mỗi mẫu.
- `X`: Ma trận đặc trưng của dữ liệu (150 x 4).
- `y_true`: Nhãn thực tế cho từng mẫu (0, 1, 2 cho ba loại hoa Iris).

### 2. Khởi tạo các tham số cho thuật toán K-Means

```python
def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])[:k]
    return X[indices]
```
- `initialize_centroids`: Hàm khởi tạo các centroid ngẫu nhiên từ dữ liệu. 
- `k`: Số lượng cụm (3 trong trường hợp này).
- `np.random.seed(42)`: Đặt hạt giống để đảm bảo kết quả tái lập được.

### 3. Tính khoảng cách và cập nhật nhãn

```python
def compute_distances(X, centroids):
    return cdist(X, centroids, 'euclidean')

def update_labels(distances):
    return np.argmin(distances, axis=1)
```
- `compute_distances`: Tính khoảng cách Euclidean từ mỗi điểm đến từng centroid.
- `update_labels`: Cập nhật nhãn cho từng điểm dữa trên khoảng cách gần nhất đến các centroid.

### 4. Cập nhật vị trí của các centroid

```python
def update_centroids(X, labels, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[labels == i]
        new_centroids[i] = points.mean(axis=0) if len(points) > 0 else np.zeros(X.shape[1])
    return new_centroids
```
- `update_centroids`: Tính toán lại vị trí của các centroid bằng cách lấy trung bình của các điểm thuộc cùng một cụm.

### 5. Chạy thuật toán K-Means

```python
def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        distances = compute_distances(X, centroids)
        labels = update_labels(distances)
        new_centroids = update_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels
```
- `kmeans`: Hàm chính thực hiện thuật toán K-Means.
- Vòng lặp sẽ tiếp tục cho đến khi không còn thay đổi centroid hoặc đạt số lần tối đa (`max_iters`).

### 6. Đánh giá kết quả

```python
def map_labels(y_true, y_kmeans):
    labels = np.zeros_like(y_kmeans)
    for i in range(k):
        mask = (y_kmeans == i)
        labels[mask] = np.bincount(y_true[mask]).argmax()
    return labels

y_kmeans_mapped = map_labels(y_true, y_kmeans)
```
- `map_labels`: Ánh xạ nhãn phân cụm dự đoán với nhãn thực tế bằng cách tìm nhãn phổ biến nhất cho mỗi cụm.

### 7. Tính toán các chỉ số đánh giá

```python
from sklearn.metrics import f1_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score

f1 = f1_score(y_true, y_kmeans_mapped, average='weighted')
rand_index = adjusted_rand_score(y_true, y_kmeans_mapped)
nmi = normalized_mutual_info_score(y_true, y_kmeans_mapped)
db_index = davies_bouldin_score(X, y_kmeans)
```
- `f1_score`: Đo lường độ chính xác của phân cụm.
- `adjusted_rand_score`: Đo độ tương đồng giữa nhãn thực tế và nhãn phân cụm.
- `normalized_mutual_info_score`: Đo lường mức độ tương đồng giữa hai phân phối nhãn.
- `davies_bouldin_score`: Đánh giá chất lượng của phân cụm dựa trên khoảng cách giữa các cụm và độ phân tán của các cụm.

### 8. In kết quả

```python
print("F1-score (weighted):", f1)
print("Adjusted Rand Index:", rand_index)
print("Normalized Mutual Information:", nmi)
print("Davies-Bouldin Index:", db_index)
```
- In ra các chỉ số đánh giá để xem mức độ chính xác của thuật toán phân cụm K-Means trên tập dữ liệu Iris.

## Kết luận

Mã này thực hiện thuật toán K-Means trên tập dữ liệu Iris và đánh giá hiệu suất của nó bằng nhiều chỉ số khác nhau. Bạn có thể sử dụng mã này như một ví dụ để hiểu rõ hơn về cách thức hoạt động của K-Means cũng như các chỉ số đánh giá chất lượng phân cụm.

## Yêu cầu

Để chạy mã này, bạn cần cài đặt các thư viện sau:

```bash
pip install numpy scikit-learn scipy
```

## Cách chạy

Sau khi cài đặt các thư viện cần thiết, bạn có thể chạy mã bằng cách thực thi tệp Python trong môi trường mà bạn đã thiết lập.

```bash
python your_script.py
```

## Tác giả

* [Tên của bạn]
