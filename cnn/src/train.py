import tensorflow as tf
from model import create_model
from data_preprocessing import get_train_val_generators

# Lấy bộ dữ liệu huấn luyện và validation
train_generator, val_generator = get_train_val_generators()

# Xây dựng mô hình
model = create_model()

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=80,  # 80 ảnh huấn luyện cho mỗi lớp
    epochs=10,           # Số lượng epoch
    validation_data=val_generator,
    validation_steps=20  # 20 ảnh kiểm tra cho mỗi lớp
)

# Lưu mô hình đã huấn luyện
model.save('animal_classifier_model.h5')

# In kết quả huấn luyện
print("Training complete. Model saved as 'animal_classifier_model.h5'")
