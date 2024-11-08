# main.py

import cv2
import os
from utils.edge_detection import sobel_edge_detection, prewitt_edge_detection, roberts_edge_detection, canny_edge_detection, gaussian_blur

# Đọc ảnh đầu vào
input_path = 'images/img.jpg'
output_dir = 'images/output'
os.makedirs(output_dir, exist_ok=True)

# Kiểm tra nếu ảnh tồn tại
if not os.path.exists(input_path):
    print("Ảnh đầu vào không tồn tại.")
    exit()

image = cv2.imread(input_path)

# Áp dụng các thuật toán phân đoạn ảnh
sobel_result = sobel_edge_detection(image)
prewitt_result = prewitt_edge_detection(image)
roberts_result = roberts_edge_detection(image)
canny_result = canny_edge_detection(image)
gaussian_result = gaussian_blur(image)

# Lưu các ảnh kết quả vào thư mục output
cv2.imwrite(os.path.join(output_dir, 'sobel_result.png'), sobel_result)
cv2.imwrite(os.path.join(output_dir, 'prewitt_result.png'), prewitt_result)
cv2.imwrite(os.path.join(output_dir, 'roberts_result.png'), roberts_result)
cv2.imwrite(os.path.join(output_dir, 'canny_result.png'), canny_result)
cv2.imwrite(os.path.join(output_dir, 'gaussian_result.png'), gaussian_result)

print("Các ảnh kết quả đã được lưu vào thư mục output.")
