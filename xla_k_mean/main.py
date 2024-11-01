import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist

# 1. Chuẩn bị dữ liệu IRIS
iris = load_iris()
X = iris.data
y_true = iris.target

# Khởi tạo các tham số cho thuật toán K-means
def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])[:k]
    return X[indices]

# Tính khoảng cách Euclidean từ mỗi điểm đến các centroid
def compute_distances(X, centroids):
    return cdist(X, centroids, 'euclidean')

# Cập nhật nhãn cho từng điểm dựa trên khoảng cách gần nhất
def update_labels(distances):
    return np.argmin(distances, axis=1)

# Tính toán lại vị trí centroid dựa trên nhãn
def update_centroids(X, labels, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[labels == i]
        new_centroids[i] = points.mean(axis=0) if len(points) > 0 else np.zeros(X.shape[1])
    return new_centroids

# Thuật toán K-means
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

# 2. Áp dụng thuật toán K-means
k = 3  # Số cụm
y_kmeans = kmeans(X, k)

# 3. Đánh giá kết quả
def map_labels(y_true, y_kmeans):
    labels = np.zeros_like(y_kmeans)
    for i in range(k):
        mask = (y_kmeans == i)
        labels[mask] = np.bincount(y_true[mask]).argmax()
    return labels

y_kmeans_mapped = map_labels(y_true, y_kmeans)

# Tính toán các chỉ số đánh giá
f1 = f1_score(y_true, y_kmeans_mapped, average='weighted')
rand_index = adjusted_rand_score(y_true, y_kmeans_mapped)
nmi = normalized_mutual_info_score(y_true, y_kmeans_mapped)
db_index = davies_bouldin_score(X, y_kmeans)

# In kết quả
print("F1-score (weighted):", f1)
print("Adjusted Rand Index:", rand_index)
print("Normalized Mutual Information:", nmi)
print("Davies-Bouldin Index:", db_index)
