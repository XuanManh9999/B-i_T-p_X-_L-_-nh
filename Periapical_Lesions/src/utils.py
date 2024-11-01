import os
import cv2
import numpy as np


def load_images_from_folder(folder, sample_size=100, img_size=(64, 64)):
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        return [], []
    images = []
    labels = []

    # Lấy danh sách file trong thư mục và trích mẫu ngẫu nhiên 100 ảnh
    image_files = os.listdir(folder)
    if len(image_files) > sample_size:
        image_files = np.random.choice(image_files, sample_size, replace=False)

    for filename in image_files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh dưới dạng grayscale
        if img is not None:
            img_resized = cv2.resize(img, img_size)  # Resize ảnh về kích thước cố định
            images.append(img_resized.flatten())  # Chuyển về mảng 1D
            # Lấy nhãn từ tên thư mục
            label = folder.split('/')[-1]
            labels.append(label)
    return np.array(images), np.array(labels)
