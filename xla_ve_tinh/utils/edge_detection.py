# utils/edge_detection.py

import cv2
import numpy as np

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(sobel)

def prewitt_edge_detection(image):
    # Đảm bảo ảnh đầu vào là kiểu float32
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Định nghĩa kernel Prewitt
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    # Áp dụng bộ lọc Prewitt
    prewitt_x = cv2.filter2D(gray, -1, kernelx)
    prewitt_y = cv2.filter2D(gray, -1, kernely)

    # Tính toán độ lớn của gradient
    prewitt = cv2.magnitude(prewitt_x, prewitt_y)

    # Chuyển đổi sang kiểu uint8 để có thể hiển thị hoặc lưu trữ
    return cv2.convertScaleAbs(prewitt)

def roberts_edge_detection(image):
    # Đảm bảo ảnh đầu vào là kiểu float32
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Định nghĩa kernel Roberts
    kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # Áp dụng bộ lọc Roberts
    roberts_x = cv2.filter2D(gray, -1, kernelx)
    roberts_y = cv2.filter2D(gray, -1, kernely)

    # Tính toán độ lớn của gradient
    roberts = cv2.magnitude(roberts_x, roberts_y)

    # Chuyển đổi sang kiểu uint8 để có thể hiển thị hoặc lưu trữ
    return cv2.convertScaleAbs(roberts)

def canny_edge_detection(image, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, threshold1, threshold2)
    return canny

def gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)
