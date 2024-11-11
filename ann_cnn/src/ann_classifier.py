import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import load_data

# Load dữ liệu
X, y = load_data('../data/train')
X_train, X_test, y_train, y_test = train_test_split(X / 255.0, y, test_size=0.2, random_state=42)

# Xây dựng mô hình ANN
ann = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Biên dịch và huấn luyện mô hình
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Dự đoán và đánh giá
y_pred_ann = (ann.predict(X_test) > 0.5).astype("int32").flatten()
print(f"ANN Accuracy: {accuracy_score(y_test, y_pred_ann) * 100:.2f}%")
