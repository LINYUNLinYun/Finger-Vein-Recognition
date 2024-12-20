import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from cv2_util import draw_reticle

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

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

# 该函数输入一个列表，其中包含许多y坐标值，将这些y坐标值绘制成折线图
def show_y_coordinate(point_list):
    x = np.arange(len(point_list))
    plt.plot(x, IMAGE_HEIGHT -  point_list[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# 该函数实现一个滑动平均滤波，根据核大小在y坐标上进行滑动平均滤波，头尾进行补0
def moving_average_filter(point_list, kernel_size = 3):
    # 核大小为奇数
    if kernel_size % 2 == 0:
        raise ValueError('The kernel size must be odd')
    # 滤波后的点集
    filtered_points = []
    # 核的一半大小
    half_kernel_size = kernel_size // 2
    # 头部补0
    for i in range(half_kernel_size):
        filtered_points.append([0,480])
    # 滑动平均滤波
    for i in range(len(point_list)):
        filtered_points.append(point_list[i])
    # 尾部补0
    for i in range(half_kernel_size):
        filtered_points.append([0,480])
    result = []
    for i in range(half_kernel_size, len(point_list) + half_kernel_size):
        sum = 0
        for j in range(kernel_size):
            sum += filtered_points[i - half_kernel_size + j][1]
        sum += filtered_points[i][1]
        for j in range(kernel_size):
            sum += filtered_points[i + half_kernel_size - j][1]
        sum/= kernel_size
        result.append([filtered_points[i][0], sum])
    return result

# 该函数创建一个k大小的滑动窗口，用来遍历一维数据找到局部最小值
def sliding_window_min(data, kernel_size = 5):
    # 滑动窗口的一半大小
    half_kernel_size = kernel_size // 2
    # 局部最小值的索引
    local_min_index = []
    # 遍历数据
    for i in range(half_kernel_size, len(data) - half_kernel_size):
        is_min = True
        for j in range(kernel_size):
            if(data[i - half_kernel_size + j] < data[i]):
                is_min = False
                break
        for j in range(kernel_size):
            if(data[i + half_kernel_size - j] < data[i]):
                is_min = False
                break
        if is_min:
            local_min_index.append(i)

    return local_min_index

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
    # 绘制轮廓点
    hand_contours_points = []
    for i in range(contours[max_contour_index].shape[0]):
        cv2.circle(img, (contours[max_contour_index][i][0][0], 
                    contours[max_contour_index][i][0][1]), 
                    5, (255, 0, 255), -1)
        hand_contours_points.append(contours[max_contour_index][i][0])
        # print(contours[max_contour_index][i][0])
    # print(hand_contours_points)
    print("cnt points num: ",len(hand_contours_points))
    hand_contours_points_dis = [(IMAGE_WIDTH/2 - point[0])**2 + (IMAGE_HEIGHT - point[1])**2 for point in hand_contours_points]
    # print("cnt points down: ", min(hand_contours_points, 
    #                                     key = lambda x: (IMAGE_WIDTH/2 - x[0])**2 + (IMAGE_HEIGHT - x[1])**2))
    root_point_index = np.argmin(hand_contours_points_dis)
    print("cnt points down index: ", np.argmin(hand_contours_points_dis))
    # 把根部点之前的所有点移动到列表末尾
    hand_contours_points = hand_contours_points[root_point_index:] + hand_contours_points[:root_point_index]
    # 对轮廓点按x由小到大排序
    # hand_contours_points = sorted(hand_contours_points, key = lambda x: x[0])
    # print("cnt points sorted", hand_contours_points)
    # 对轮廓点的y坐标进行滑动平均滤波
    # hand_contours_points = np.array(hand_contours_points)
    hand_contours_points = moving_average_filter(hand_contours_points,15)
    hand_cnt_points_y = IMAGE_HEIGHT - np.array(hand_contours_points)[:,-1]
    for i in range(hand_cnt_points_y.shape[0]):
        print(f"- NO.{i}: {hand_cnt_points_y[i]}")
    local_min_list = sliding_window_min(hand_cnt_points_y,9)
    print("local min points: ",local_min_list)
    for l_p in local_min_list:
        print(f"local min point: ({hand_contours_points[l_p][0]},{hand_contours_points[l_p][1]})")
        print(type(hand_contours_points[l_p]))
        # draw_reticle(img, hand_contours_points[l_p][0],hand_contours_points[l_p][1], label_color=(0, 0, 255))
        cv2.circle(img, (hand_contours_points[l_p]), 10, (0, 0, 255), -1)
    # for p in hand_cnt_points_y:
    #     print(p)

    show_y_coordinate(np.array(hand_contours_points))

    
    
    # 手部的轮廓集合
    # 绘制凸包缺陷点，效果不好，废弃
    # print(hull.shape)
    # hull = cv2.convexHull(contours[max_contour_index], returnPoints=False)
    # defects = cv2.convexityDefects(contours[max_contour_index], hull)
    # print(defects.shape)
    # for i in range(defects.shape[0]):
    #     # 起点、终点、凸包缺陷点(最远点)、到最远点的近似距离
    #     s, e, f, d = defects[i, 0]
    #     start = tuple(contours[max_contour_index][s][0])
    #     end = tuple(contours[max_contour_index][e][0])
    #     far = tuple(contours[max_contour_index][f][0])
    #     cv2.line(img, start, end, [0, 255, 0], 2)
    #     cv2.circle(img, far, 25, [0, 0, 255], -1)
        
    
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
    img = find_corners('PV1_binary/134/4.bmp', True)
    key = 0
    while True:
        cv2.imshow('contours', img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    