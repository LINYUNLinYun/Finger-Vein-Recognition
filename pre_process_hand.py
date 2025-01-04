import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from cv2_util import draw_reticle
import gc

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 该函数用于将分散于PV1、PV2和PPPV文件夹下的图片进行聚合，方便后续处理
def get_all_images():
    # 获取当前目录
    current_path = os.getcwd()
    # 获取当前目录下的PV1、PV2和PPPV文件夹
    pv1_path = os.path.join(current_path, 'PV1')
    pv2_path = os.path.join(current_path, 'PV2')
    pppv_path = os.path.join(current_path, 'PPPV')
    # 获取PV1、PV2和PPPV文件夹下的所有样本文件夹
    dirs = os.listdir(pv1_path)
    samples_num = 10
    PV1_img_path = []
    PV2_img_path = []
    PPPV_img_path = []
    # 检查它是不是.bmp文件 否则报错
    for dir in dirs:
        # print(file)
        for file in os.listdir(os.path.join(pv1_path, dir)):
            if file[-4:] != '.bmp':
                raise ValueError('The file is not a .bmp file')
            # 获取文件的路径
            file_path = os.path.join(pv1_path, dir, file)
            PV1_img_path.append(file_path)
    for dir in dirs:
        # print(file)
        for file in os.listdir(os.path.join(pv2_path, dir)):
            if file[-4:] != '.bmp':
                raise ValueError('The file is not a .bmp file')
            # 获取文件的路径
            file_path = os.path.join(pv2_path, dir, file)
            PV2_img_path.append(file_path)
    for dir in dirs:
        # print(file)
        for file in os.listdir(os.path.join(pppv_path, dir, 'PV')):
            if file[-4:] != '.bmp':
                raise ValueError('The file is not a .bmp file')
            # 获取文件的路径
            file_path = os.path.join(pppv_path, dir, 'PV' ,file)
            PPPV_img_path.append(file_path)
    all_imgs_path = []
    for i in range(samples_num):
        temp = []
        temp = temp + PV1_img_path[10*i:10*i+10]
        temp = temp + PV2_img_path[10*i:10*i+10]
        temp = temp + PPPV_img_path[10*i:10*i+10]
        all_imgs_path.append(temp)
    # print(all_imgs_path)
    print("all_nums",len(all_imgs_path))
    for i in range(len(all_imgs_path)):
        for j in range(len(all_imgs_path[i])):
            img = cv2.imread(all_imgs_path[i][j])
            # 如果路径不存在则创建路径
            if not os.path.exists(os.path.join(current_path,'samples',str(i))):
                os.makedirs(os.path.join(current_path,'samples',str(i)))
            cv2.imwrite(os.path.join(current_path,'samples',str(i),str(j)+'.bmp'), img)
            # cv2.imshow('image', img)
            # # 等待键盘输入
            # cv2.waitKey(0)
            # # 关闭窗口
            # cv2.destroyAllWindows()


# 该函数将读取当前目录下的文件夹，并将里面的图片显示出来
def save_bin_images(dir_path='target_path'):
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


# 该函数用于获取一个轮廓的最大内接圆，并把这个圆绘制出来
def get_max_inscribed_circle(binary, contour,img,draw_enable=False):
  
    # 计算到轮廓的距离
    raw_dist = np.empty(binary.shape, dtype=np.float32)
    for i in range(binary.shape[0]):
        for j in range(binary.shape[1]):
            raw_dist[i, j] = cv2.pointPolygonTest(contour, (j, i), True)
    
    # 获取最大值即内接圆半径，中心点坐标
    minVal, maxVal, _, maxDistPt = cv2.minMaxLoc(raw_dist)
    minVal = abs(minVal)
    maxVal = abs(maxVal)
    
    # 画出最大内接圆
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    radius = int(maxVal)
    center_of_circle = maxDistPt
    if draw_enable:
        cv2.circle(result, maxDistPt, radius, (0, 255, 0), 2, 1, 0)
        cv2.imshow('Maximum inscribed circle', result)
        cv2.waitKey(0)
    return maxDistPt, radius
    
# 该函数用于将一个img进行旋转，使用角度为angle，旋转过程中对图像进行扩充，draw_enable用于控制是否绘制旋转后的图像
def rotate_image(img, angle,center, draw_enable=False):
    # 获取img的中心点
    center = (img.shape[1] // 2, img.shape[0] // 2)
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 旋转图像
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),cv2.INTER_CUBIC)
    # 绘制旋转后的图像
    if draw_enable:
        cv2.imshow('Rotated image', rotated)
        cv2.waitKey(0)  
    # cv2.imshow('Rotated image', rotated)
    # cv2.waitKey(0)
    return rotated

# 该函数用于截取一个图像的一部分，输入为图像img，中心点center，宽度width，高度height
def crop_image(img, center, width, height):
    # 获取左上角点
    x = center[0] - width // 2
    y = center[1] - height // 2
    # 截取图像
    cropped = img[y:y + height, x:x + width]
    return cropped

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
        result.append([filtered_points[i][0], round(sum)])
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

# 该函数用于将读取的图片转化为二值
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

def find_intrest_roi(binary,img,draw_enable = False):
    
    # 找轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 
    contours_area = []
    for contour in contours:
        # 获取轮廓的面积
        contours_area.append(cv2.contourArea(contour))
    # print(contours_area)
    max_contour_index = np.argmax(contours_area)
    # 寻找最小外接矩形
    rect = cv2.minAreaRect(contours[max_contour_index])
    print(rect)
    # 绘制最小外接矩形
    box = cv2.boxPoints(rect)
    # box = np.intp(box)
    print(box)
    if draw_enable:
        # 这一步不影响后面的画图，但是可以保证四个角点坐标为顺时针
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box,4-startidx,0)
        # 在原图上画出预测的外接矩形
        box = box.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(img,[box],True,(0,255,0),10)
        cv2.imshow('roi', img)
        cv2.waitKey(0)



# 该函数用于找到手部的轮廓，找到手部的根部点，找到手部的中心点，找到手部的角度等，提取出掌静脉的ROI
def find_roi(binary_path = 'PV1_binary/131/1.bmp',draw_enable=False):
    img_path = binary_path.replace('_binary', '')
    # 如果图像大小超过640*480则缩小
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (640, 480))
    IMAGE_HEIGHT = img.shape[0]
    IMAGE_WIDTH = img.shape[1]
    img_orin = img.copy()
    binary = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    # binary = cv2.resize(binary, (640, 480))
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
    # if draw_enable:
    #     cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    #     cv2.imshow('rrr', img)
    #     cv2.waitKey(0)
    # 截取得到roi
    hand_roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    hand_roi_binary = binary[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    IMAGE_HEIGHT = hand_roi.shape[0]
    IMAGE_WIDTH =  hand_roi.shape[1]
    cv2.imshow('hand_roi', hand_roi_binary)
    cv2.waitKey(0)
    # 找轮廓
    contours, hierarchy = cv2.findContours(hand_roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_area = []
    for contour in contours:
        # 获取轮廓的面积
        contours_area.append(cv2.contourArea(contour))
    # print(contours_area)
    max_contour_index = np.argmax(contours_area)

    
    hand_contours_points = []
    for i in range(contours[max_contour_index].shape[0]): 
        hand_contours_points.append(contours[max_contour_index][i][0])
        # print(contours[max_contour_index][i][0])
    # print(hand_contours_points)
    # print("cnt points num: ",len(hand_contours_points))
    hand_contours_points_dis = [(IMAGE_WIDTH/2 - point[0])**2 + (IMAGE_HEIGHT - point[1])**2 for point in hand_contours_points]
    # print("cnt points down: ", min(hand_contours_points, 
    #                                     key = lambda x: (IMAGE_WIDTH/2 - x[0])**2 + (IMAGE_HEIGHT - x[1])**2))
    root_point_index = np.argmin(hand_contours_points_dis)
    # print("cnt points down index: ", np.argmin(hand_contours_points_dis))
    # 把根部点之前的所有点移动到列表末尾
    hand_contours_points = hand_contours_points[root_point_index:] + hand_contours_points[:root_point_index]
    # 对轮廓点按x由小到大排序
    # hand_contours_points = sorted(hand_contours_points, key = lambda x: x[0])
    # print("cnt points sorted", hand_contours_points)
    # 对轮廓点的y坐标进行滑动平均滤波
    # hand_contours_points = np.array(hand_contours_points)
    hand_contours_points_origin = hand_contours_points
    hand_cnt_points_y = IMAGE_HEIGHT - np.array(hand_contours_points)[:,-1]
    # hand_contours_points = moving_average_filter(hand_contours_points,9)
    # for i in range(hand_cnt_points_y.shape[0]):
        # print(f"- NO.{i}: {hand_cnt_points_y[i]}")
    local_min_list = sliding_window_min(hand_cnt_points_y,9)
    #根据hand_contours_points_origin[local_min_list][1]的大小对local_min_list进行由大到小排序
    local_min_list = sorted(local_min_list, key = lambda x: IMAGE_HEIGHT-hand_contours_points_origin[x][1],reverse=True)
    print("len of local_min_list: ",len(local_min_list))
    local_min_list = local_min_list[:4]
    # 求剩下的点的平均值
    fingers_center_x = 0
    fingers_center_y = 0
    for index in local_min_list:
        fingers_center_x += hand_contours_points_origin[index][0]
        fingers_center_y += hand_contours_points_origin[index][1]
    fingers_center_x /= len(local_min_list)
    fingers_center_y /= len(local_min_list)
    fingers_center_x = int(fingers_center_x)
    fingers_center_y = int(fingers_center_y)
    if draw_enable:
        draw_reticle(hand_roi, fingers_center_x, fingers_center_y, label_color=(0, 0, 255))
        cv2.imshow('hand_roi', hand_roi)
        cv2.waitKey(0)
    
    # cv2.rotatedRectangleIntersection()
    # show_y_coordinate(np.array(hand_contours_points))
    center,r = get_max_inscribed_circle(hand_roi_binary, contours[max_contour_index],img,False)
    # print(center,r)
    vector_hand2fingers = np.array([(fingers_center_x - center[0]), -(fingers_center_y - center[1])])
    print("vector_hand2fingers: ",vector_hand2fingers)
    angle = 90 - np.arctan2(vector_hand2fingers[1],vector_hand2fingers[0]) * 180 / np.pi
    # 旋转图像
    
    rotated_img = rotate_image(hand_roi, angle,(center[0],center[1]), False)
    # 如果不是灰度图则转化为灰度图
    if len(rotated_img.shape) == 3:
        rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
    roi = crop_image(rotated_img, center, int(r*1), int(r*1))
    roi = clahe_image(roi)
    roi = cv2.resize(roi, (224, 224))
    # cv2.imshow('roi', roi)
    gc.collect()
    
    return roi

def clahe_image(image):
    """
    限制对比度自适应直方图均衡化(CLAHE)函数
    :param image: 输入的图像（灰度图像）
    :return: CLAHE处理后的图像
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def get_train_hand_dataset():
    current_path = os.getcwd()
    binary_samples_path = os.path.join(current_path, 'samples_binary')
    for dir in os.listdir(binary_samples_path):
        print(dir)
        for file in os.listdir(os.path.join(binary_samples_path, dir)):
            print(file) 
            # img = cv2.imread(os.path.join(binary_samples_path, dir, file))
            img_roi = find_roi(os.path.join(binary_samples_path, dir, file), False)

            if not os.path.exists(os.path.join(current_path,'train_hand',dir)):
                os.makedirs(os.path.join(current_path,'train_hand',dir))
            cv2.imwrite(os.path.join(current_path,'train_hand',dir,file), img_roi)

if __name__ == '__main__':
    
    get_train_hand_dataset()
    # img_roi = find_roi('samples_binary/0/26.bmp', True)
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
    