import os

# Xác định đường dẫn tuyệt đối của thư mục 'data'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Xuan manh check BASE_DIR", BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'train')
def load_data(data_dir=DATA_DIR, img_size=(64, 64)):
    labels = []
    data = []
    
    for label, sub_dir in enumerate(['cats', 'dogs']):
        path = os.path.join(data_dir, sub_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {path}")
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append(img.flatten())
            labels.append(label)
    
    return np.array(data), np.array(labels)
