# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from utils import load_images_from_folder

# # Danh sách các thư mục chứa ảnh
# class_dirs = ['Augmentation_JPG_Images', 'Image_Annots', 'Original_JPG_Images']

# # Khởi tạo danh sách cho dữ liệu
# X = []
# y = []

# # Lấy 100 ảnh từ mỗi thư mục
# for class_dir in class_dirs:
#     X_class, y_class = load_images_from_folder(class_dir, sample_size=100)
#     X.append(X_class)
#     y.append(y_class)

# # Ghép dữ liệu lại với nhau
# X = np.concatenate(X)
# y = np.concatenate(y)

# # Chia dữ liệu thành tập train và test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Chia theo tỉ lệ 80-20

# # Khởi tạo và đánh giá mô hình
# results = []

# # SVM
# svm_model = SVC(kernel='linear')
# svm_model.fit(X_train, y_train)
# y_pred_svm = svm_model.predict(X_test)

# # KNN
# knn_model = KNeighborsClassifier(n_neighbors=5)
# knn_model.fit(X_train, y_train)
# y_pred_knn = knn_model.predict(X_test)

# # Đánh giá hiệu suất
# for model_name, y_pred in zip(['SVM', 'KNN'], [y_pred_svm, y_pred_knn]):
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')

#     # Lưu kết quả
#     results.append({
#         'Model': model_name,
#         'Accuracy': accuracy,
#         'Precision': precision,
#         'Recall': recall,
#         'F1 Score': f1
#     })

# # In kết quả
# for result in results:
#     print(
#         f"{result['Model']} - Accuracy: {result['Accuracy']}, Precision: {result['Precision']}, Recall: {result['Recall']}, F1 Score: {result['F1 Score']}")

import os
import cv2
from skimage.feature import hog
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Định dạng thư mục dữ liệu
image_dir = './Augmentation_JPG_Images/'  # Thay đổi nếu cần
labels = []  # Thêm nhãn của ảnh ở đây
features = []

# Trích xuất đặc trưng từ ảnh
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Resize ảnh để đồng nhất kích thước
    image = cv2.resize(image, (128, 128))  # Giả sử ảnh đều có thể resize về 128x128
    # Trích xuất HOG feature
    hog_feature = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    features.append(hog_feature)
    # Thêm nhãn vào (cần ánh xạ tên ảnh vào nhãn)
    labels.append(0 if "class0" in img_name else 1)  # Thay "class0" và "class1" bằng tên lớp thực tế của bạn

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Naive Bayes -> BASE -> 
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))

# CART (Gini Index)
cart_model = DecisionTreeClassifier(criterion="gini")
cart_model.fit(X_train, y_train)
cart_predictions = cart_model.predict(X_test)
print("CART (Gini Index) Accuracy:", accuracy_score(y_test, cart_predictions))

# Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
nn_model.fit(X_train, y_train)
nn_predictions = nn_model.predict(X_test)
print("Neural Network Accuracy:", accuracy_score(y_test, nn_predictions))
