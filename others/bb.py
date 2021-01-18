import numpy as np
import cv2
import sys
import math


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = int(round(i_filtered))


def bilateral_filter_own(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image


if __name__ == "__main__":
    src = cv2.imread('../RAWPIC/5.png', cv2.IMREAD_COLOR)
    canny = cv2.bilateralFilter(src, 30, 30, 20)
    cv2.imwrite('edges.png', canny)
    # src_gray = cv2.imread('../PICS/1.jpg', cv2.IMREAD_GRAYSCALE)
    # filtered_image_OpenCV = cv2.bilateralFilter(src, 20, 20, 20.0)
    # cv2.imwrite("original_image_grayscale.png", src_gray)
    # cv2.imwrite("original_image_color.png", src)
    # cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
    # filtered_image_own = bilateral_filter_own(src, 5, 12.0, 16.0)
    # cv2.imwrite("filtered_image_own.png", filtered_image_own)







