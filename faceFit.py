import numpy as np
from vec2D import *
import matplotlib.pyplot as plt
from featureExtraction import *

radius: int
image: np.ndarray
circle: np.ndarray


# 确定圆内像素
def create_circle():
    global circle, radius
    length = 2 * radius + 1
    circle = np.ones((length, length), 'uint8')
    for r in range(length):
        for c in range(length):
            if vec_dis((r, c), (radius, radius)) > radius:
                circle[r, c] = 0


# 像素坐标计算
def coordinate(direct: tuple, coord: tuple):
    global radius
    center = (radius, radius)
    a = pow(radius, 2) - pow(vec_abs(vec_sub(coord, center)), 2)
    b = a + pow(vec_abs(direct), 2)
    return vec_sub(coord, vec_times(pow(a / b, 2), direct))


# 从四个相邻原像素中计算新的像素值
def new_pixel_value(coord: tuple):
    coord_int = (int(coord[0]), int(coord[1]))
    d = vec_sub(coord, coord_int)
    # 生成一个2*2的权值矩阵
    weight = np.ones((2, 2), 'float32')
    weight[0: 1, :] *= 1 - d[0]
    weight[1: 2, :] *= d[0]
    weight[:, 0: 1] *= 1 - d[1]
    weight[:, 1: 2] *= d[1]
    weight /= weight.sum()
    # 取原像素点附近的四个像素值
    pixel = np.zeros(3, 'float32')
    for i in range(image.shape[2]):
        piece = image[coord_int[0]: coord_int[0] + 2, coord_int[1]: coord_int[1] + 2, i: i + 1].reshape((2, 2))
        pixel[i] = (weight * piece).sum()
    return pixel


# 交互式局部图像平移处理
def local_translation(coord_center: tuple, coord_direct: tuple):
    global image, radius, circle
    length = 2 * radius * 1
    coord_origin = vec_to_int(vec_sub(coord_center, (radius, radius)))  # 相对坐标原点
    # 复制一个局部缓冲区存储新的像素值
    buffer = image[coord_origin[0]: coord_origin[0] + length, coord_origin[1]: coord_origin[1] + length, :].copy()
    for r in range(length):
        for c in range(length):
            # 遍历区域中每一个像素点
            coord = (r, c)
            if circle[coord] == 1:  # 如果是圆内像素点
                # 调用coordinate计算原像素点的相对坐标，然后转化成绝对坐标
                coord_old = vec_add(coord_origin, coordinate(coord_direct, coord))
                # 调用new_pixel_value计算新像素值，并且装载到缓冲区的相应位置
                buffer[coord] = new_pixel_value(coord_old)
    # 将缓冲区内的新值覆盖到原图像上
    image[coord_origin[0]: coord_origin[0] + length, coord_origin[1]: coord_origin[1] + length, :] = buffer


# img: (row, column, bgr)
# feature: 顺序正确的68个脸部轮廓特征坐标, 形式: [[r1 c1] [r2 c2] [r3 c3] ...], r表示行数, c表示列数
# ignore: 忽略的脸部轮廓特征点集合
# rad: 局部图像平移算法的半径
# distance: 局部图像平移算法的移动距离
def face_thin_auto(img: np.ndarray, feature: np.ndarray, ignore, rad, distance):
    global image, radius
    image = img
    radius = rad
    create_circle()
    coord_nose = tuple(feature[30])  # 鼻尖坐标，用于标记方向
    face = [tuple(feature[i]) for i in range(17)]  # 脸部轮廓坐标
    for i in range(1, 16):
        if set(ignore).__contains__(i):
            continue
        coord_center = face[i]
        coord_l = face[i - 1]
        coord_r = face[i + 1]
        # 先计算平移方向向量
        coord_direct = vec_times(distance, vec_normal(coord_l, coord_r, coord_nose))
        # 调用local_translation进行局部平移
        local_translation(coord_center, coord_direct)
    return image


if __name__ == '__main__':
    pic_name = input('picture name: ')
    image2 = plt.imread(pic_name)[:, :, 0: 3].astype('float32')
    featureList_x, featureList_y = extract_feature(pic_name)
    for i in range(len(featureList_x)):
        r_lst = featureList_x[i]
        c_lst = featureList_y[i]
        ignore = {0, 1, 2, 8, 14, 15, 16}
        feature = np.dstack((r_lst, c_lst)).reshape((68, 2))
        image2 = face_thin_auto(image2, feature, ignore, rad=40, distance=80)
        plt.imsave('thin_'+pic_name, image2.astype('uint8'))
