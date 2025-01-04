import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from cv2_util import draw_reticle
import gc

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 该函数用于获取当前目录下的所有图片路径
def get_all_files(dir_path):
    # current_path = os.getcwd()
    dirs = os.listdir(dir_path)
    files_path = []
    for dir in dirs:
        pic_dir = os.path.join(dir_path, dir)
        for j in os.listdir(pic_dir):
            if j.endswith('.bmp'):
                files_path.append(os.path.join(pic_dir, j))

    return files_path

# 该函数用于输出图片中心20*20的灰度值
def get_center_pixel_value(img):
    # 获取图片中心20*20的灰度值
    center_x = int(img.shape[1] / 2)
    center_y = int(img.shape[0] / 2)
    center = img[center_y - 10:center_y + 10, center_x - 10:center_x + 10]
    print(center)
    return center

# 该函数用于提取roi区域
def get_finger_vein_roi(img,draw_enable=False): 
    # get_center_pixel_value(img)
    # 二值化处理
    _, binary = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 进行腐蚀和膨胀操作
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # binary = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    # 找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_area = []
    for contour in contours:
        # 获取轮廓的面积
        contours_area.append(cv2.contourArea(contour))
    # print(contours_area)
    max_contour_index = np.argmax(contours_area)
    # 找到外接矩形
    rect = cv2.boundingRect(contours[max_contour_index])
    # 绘制
    M = cv2.moments(contours[max_contour_index])
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    left_x = rect[0]
    right_x = rect[0] + rect[2]
    rect_center = (rect[0] + int(rect[2]/2), rect[1] + int(rect[3]/2))
    # 绘制roi区域
    if rect_center[1] < center_y:
        center_y += int((center_y - rect_center[1])*0.75)
    else:
        center_y -= int((rect_center[1] - center_y)*0.75)
    up_y = center_y - int(rect[3]*0.6 / 2)
    down_y = center_y + int(rect[3]*0.6 / 2)
    roi = img[up_y:down_y, left_x:right_x]
    if draw_enable:
        # 绘制轮廓的质心
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        cv2.rectangle(img, (left_x, up_y), (right_x, down_y), (255, 0, 0), 3)
        cv2.imshow('rrr', img)
        cv2.waitKey(0)
    
        # 显示
        cv2.imshow('binary', binary)
        cv2.waitKey(0)
    return roi
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
def get_train_finger_dataset(input_path, output_path):
    # 获取当前目录下的所有图片路径
    all_files = get_all_files(input_path)
    # 遍历所有图片
    for i, file in enumerate(all_files):
        # print(file)
        back = file.split('\\')[-2]
        name = file.split('\\')[-1]
        if not os.path.exists(os.path.join(output_path, back)):
            os.makedirs(os.path.join(output_path, back))
        
        # 读取图片
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # 提取roi区域
        roi = get_finger_vein_roi(img,False)
        # 图像增强
        roi = clahe_image(roi)
        # 保存图片
        cv2.imwrite(os.path.join(output_path, back, name), roi)
        # 释放内存
        gc.collect()
        # break

import cv2
import matplotlib.pyplot as plt
import numpy as np


def global_histogram_equalization(image):
    """
    全局直方图均衡化函数
    :param image: 输入的图像（灰度图像）
    :return: 直方图均衡化后的图像
    """
    return cv2.equalizeHist(image)


def clahe_image(image):
    """
    限制对比度自适应直方图均衡化(CLAHE)函数
    :param image: 输入的图像（灰度图像）
    :return: CLAHE处理后的图像
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def retinex_image(image, sigma=150):
    """
    Retinex算法实现图像增强（单尺度Retinex）
    :param image: 输入的图像（灰度图像）
    :param sigma: 高斯模糊的标准差，控制增强效果
    :return: Retinex增强后的图像
    """
    image = image.astype(np.float32) / 255.0
    log_image = np.log10(image + 1e-10)
    blur = cv2.GaussianBlur(log_image, (0, 0), sigma)
    result = np.uint8((np.power(10, log_image - blur) * 255).clip(0, 255))
    return result


def gray_normalization(image):
    """
    灰度归一化函数
    :param image: 输入的图像（灰度图像）
    :return: 灰度归一化后的图像
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return image
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)


# # 读取示例图像（这里以灰度图像为例，你可以替换为自己的图像路径）
# image_path = 'train_finger/6/3.bmp'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # 分别进行图像增强
# global_equalized_image = global_histogram_equalization(image)
# clahe_enhanced_image = clahe_image(image)
# retinex_enhanced_image = retinex_image(image)
# gray_normalized_image = gray_normalization(image)

# # 设置图像显示的标题
# titles = ['Original Image', 'Global Histogram Equalization', 'CLAHE', 'Retinex', 'Gray Normalization']
# images = [image, global_equalized_image, clahe_enhanced_image, retinex_enhanced_image, gray_normalized_image]

# # 使用matplotlib显示图像
# plt.figure(figsize=(10, 8))
# for i in range(len(images)):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(images[i], cmap='gray')
#     plt.title(titles[i])
#     plt.axis('off')
# plt.show()

if __name__ == '__main__':
    
    # get_train_finger_dataset()
    all_files = get_all_files('finger vein')
    # print(all_files)
    # get_finger_vein_roi(cv2.imread(all_files[58], cv2.IMREAD_GRAYSCALE),True)
    get_train_finger_dataset('finger vein', 'train_finger')
    # img_roi = find_roi('samples_binary/0/4.bmp', True)
    # binary_path = 'samples_binary/9/1.bmp'
    # binary = cv2.imread('samples_binary/2/1.bmp',cv2.IMREAD_GRAYSCALE)
    # img_path = binary_path.replace('_binary', '')
    # img = cv2.imread(img_path)
    # img_roi= find_roi(binary_path, True)
    # # find_intrest_roi(binary,img,True)
    # key = 0
    # while True:
        
    #     cv2.imshow('roi_', img_roi)
    #     key = cv2.waitKey(0)
    #     if key == 27:
    #         break
    