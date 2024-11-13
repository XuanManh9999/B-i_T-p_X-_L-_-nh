import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_val_generators():
    # Tạo đối tượng ImageDataGenerator cho tiền xử lý dữ liệu
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Đọc dữ liệu từ thư mục
    train_generator = train_datagen.flow_from_directory(
        'data/train/',   # Đường dẫn đến dữ liệu huấn luyện
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )
    
    val_generator = validation_datagen.flow_from_directory(
        'data/validation/',  # Đường dẫn đến dữ liệu validation
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )
    
    return train_generator, val_generator
