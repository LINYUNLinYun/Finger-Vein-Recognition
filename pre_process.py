import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# 该函数将读取当前目录下的文件夹，并将里面的图片显示出来
def save_bin_images(dir_path='PV1'):
    # 获取当前目录
    current_path = os.getcwd()
    # 获取当前目录下的PV文件夹
    pv_path = os.path.join(current_path, dir_path)
    # 获取PV文件夹下的所有样本文件夹
    dirs = os.listdir(pv_path)
    # 检查它是不是.bmp文件 否则报错
    for dir in dirs:
        # print(file)
        for file in os.listdir(os.path.join(pv_path, dir)):
            if file[-4:] != '.bmp':
                raise ValueError('The file is not a .bmp file')
            # 获取文件的路径
            file_path = os.path.join(pv_path, dir, file)
            # 读取图片
            img = cv2.imread(file_path)
            print(img.shape)
            # 显示图片
            binary = bgr2binary(img)

            # 保存图片 
            output_path = os.path.join(current_path,dir_path+'_binary',dir, file)
            # 如果路径不存在则创建路径
            if not os.path.exists(os.path.join(current_path,dir_path+'_binary',dir)):
                os.makedirs(os.path.join(current_path,dir_path+'_binary',dir))
            # print(output_path)
            cv2.imwrite(output_path, binary)

            # cv2.imshow('image', binary)
            # # 等待键盘输入
            # cv2.waitKey(0)
            # # 关闭窗口
            # cv2.destroyAllWindows()
        # if file[-4:] != '.bmp':
        #     raise ValueError('The file is not a .bmp file')
    # 遍历文件
    # for file in files:

# 该函数将读取当前目录下的文件夹，并将里面的图片显示出来
def save_bin_images_for_PPPV(dir_path='PPPV'):
    # 获取当前目录
    current_path = os.getcwd()
    # 获取当前目录下的PV文件夹
    pv_path = os.path.join(current_path, dir_path)
    # 获取PV文件夹下的所有样本文件夹
    dirs = os.listdir(pv_path)
    # 检查它是不是.bmp文件 否则报错
    for dir in dirs:
        # print(os.path.join(pv_path, dir, 'PV'))
        # continue
        for file in os.listdir(os.path.join(pv_path, dir, 'PV')):
            # print(file)
            if file[-4:] != '.bmp':
                raise ValueError('The file is not a .bmp file')
            # 获取文件的路径
            file_path = os.path.join(pv_path, dir, 'PV', file)
            # 读取图片
            img = cv2.imread(file_path)
            # print(img.shape)
            # 显示图片
            binary = bgr2binary(img)

            # 保存图片 
            output_path = os.path.join(current_path,dir_path+'_binary',dir, file)
            # 如果路径不存在则创建路径
            if not os.path.exists(os.path.join(current_path,dir_path+'_binary',dir)):
                os.makedirs(os.path.join(current_path,dir_path+'_binary',dir))
            # print(output_path)
            cv2.imwrite(output_path, binary)

# 该函数获取一个灰度图的所有像素点的值，范围在0-255，统计后绘成直方图显示出来
# 必要使用到可能matplotlib包
def show_gray_histogram(gray_image):
    # 获取灰度图的所有像素点的值
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # 输出直方图的众数，中位数，四分之一数等
    # print('直方图的众数:', cv2.mode(gray_image))
    # print('直方图的中位数:', np.median(gray_image))
    
    # 绘制直方图
    plt.plot(hist, color='gray')
    plt.xlabel('灰度值')
    plt.ylabel('像素点')
    plt.show()

# 该函数用于将读取的图片转化为灰度图，在利用otsu算法进行二值化
def bgr2binary(img):
    # 将图片转化为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 对灰度图做一个高斯滤波
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # 对灰度图做一个均值滤波
    gray = cv2.blur(gray, (7, 7))

    # show_gray_histogram(gray)
    # 二值化
    ret, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY )
    # 先腐蚀后膨胀，核大小为7*7
    kernel = np.ones((7, 7), np.uint8)
    # binary = cv2.dilate(binary, kernel)
    binary = cv2.erode(binary, kernel)
    binary = cv2.dilate(binary, kernel)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary

# 该函数用于寻找二值图中的角点
def find_corners(binary_path = 'PV1_binary/131/1.bmp',draw_enable=False):
    img_path = binary_path.replace('_binary', '')
    img = cv2.imread(img_path)
    binary = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    # 找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_area = []
    for contour in contours:
        # 获取轮廓的面积
        contours_area.append(cv2.contourArea(contour))
    print(contours_area)
    max_contour_index = np.argmax(contours_area)
    # 找到凸包角点
    hull=cv2.convexHull(contours[max_contour_index])
    # 绘制凸包角点
    cv2.polylines(img, [hull], True, (0, 0, 255), 2)
    # print(contours[max_contour_index].shape)
    for i in range(contours[max_contour_index].shape[0]):
        cv2.circle(img, (contours[max_contour_index][i][0][0], contours[max_contour_index][i][0][1]), 5, (255, 255, 2), -1)
    # 绘制凸包缺陷点
    # defects = cv2.convexityDefects(contours[max_contour_index], hull)
    # print(defects)

    # for i in range(defects.shape[0]):
    #     s, e, f, d = defects[i, 0]
    #     start = tuple(contours[0][s][0])
    #     end = tuple(contours[0][e][0])
    #     far = tuple(contours[0][f][0])
    #     cv2.line(img, start, end, [0, 255, 0], 2)
    #     cv2.circle(img, far, 5, [0, 0, 255], -1)
        
    
    # contour = max(contours, key= lambda x: cv2.contourArea(x))
    # print(len(contours))
    # for contour in contours:
    #     # 获取轮廓的面积
    #     area = cv2.contourArea(contour)
        
    if draw_enable:
        # 画出轮廓
        cv2.drawContours(img, contours,max_contour_index, (0, 255, 0), 3)
    return img

if __name__ == '__main__':
    # save_bin_images('PV1')
    # save_bin_images('PV2')
    # save_bin_images_for_PPPV('PPPV')
    # binary = cv2.imread('PV1_binary/131/1.bmp',cv2.IMREAD_GRAYSCALE)
    img = find_corners('PV1_binary/131/1.bmp', True)
    key = 0
    while True:
        cv2.imshow('contours', img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    