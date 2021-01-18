import cv2
import math


def beta_map(value, beta):
    return math.log((value * (beta - 1.0) + 1.0))/math.log(beta)


def change_lightness(src):
    row_num = src.shape[0]
    col_num = src.shape[1]

    for i in range(row_num):
        for j in range(col_num):
            bf_map = float(src.item(i, j, 2)) / 256.0
            af_map = int(beta_map(bf_map, 2) * 256.0)
            src.itemset((i, j, 2), af_map)


if __name__ == '__main__':

    src = cv2.imread('original_face.png', cv2.IMREAD_COLOR)
    print(type(src))
    # src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    # change_lightness(src_hsv)
    # show = cv2.cvtColor(src_hsv, cv2.COLOR_HSV2BGR)
    # cv2.imwrite('white.png', show)

