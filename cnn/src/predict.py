import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

def predict_image(img_path):
    # Tải mô hình đã huấn luyện
    model = tf.keras.models.load_model('animal_classifier_model.h5')
    
    # Tiền xử lý ảnh đầu vào
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Dự đoán
    prediction = model.predict(img_array)
    
    # In kết quả dự đoán
    if prediction[0] > 0.5:
        print("Dự đoán: Chó")
    else:
        print("Dự đoán: Mèo")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Vui lòng cung cấp đường dẫn tới ảnh cần dự đoán.")
    else:
        img_path = sys.argv[1]
        predict_image(img_path)
