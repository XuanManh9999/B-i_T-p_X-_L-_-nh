from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from preprocess import load_data

# Load dữ liệu
X, y = load_data('../data/train')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred_svm = svm.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm) * 100:.2f}%")
