import cv2
import numpy
import math


def beta_map(value, beta):
    return math.log((value * (beta - 1.0) + 1.0))/math.log(beta)


def change_lightness(src, beta):
    row_num = src.shape[0]
    col_num = src.shape[1]

    for i in range(row_num):
        for j in range(col_num):
            bf_map = float(src.item(i, j, 2)) / 256.0
            af_map = int(beta_map(bf_map, beta) * 256.0)
            src.itemset((i, j, 2), af_map)


if __name__ == '__main__':
    name = input('picture name: ')

    if name.find('.jpg') == 1 or name.find('.png') == 1:
        src = cv2.imread('../RAWPIC/' + name, cv2.IMREAD_COLOR)
        # g = cv2.GaussianBlur(src, (15, 15), 7)

        # # 磨皮
        after_filtered = cv2.bilateralFilter(src, 30, 35, 35)
        #
        # # # 转化为hsv编码
        src_hsv = cv2.cvtColor(after_filtered, cv2.COLOR_BGR2HSV)
        #
        # # 美白
        change_lightness(src_hsv, 4)
        after_white = cv2.cvtColor(src_hsv, cv2.COLOR_HSV2BGR)
        #
        show = after_white
        #
        # k = 1
        # # 锐化
        # for i in range(k):
        #     low_freq = cv2.GaussianBlur(show, (5, 5), 1.0)
        #     template = cv2.subtract(show, low_freq)
        #     src = cv2.add(show, template)
        # cv2.imwrite('../PSPIC/better_' + name, show)
        cv2.imwrite('./4_mb.png', show)

    else:
        print('picture format must be jpg or png')
        exit(1)

