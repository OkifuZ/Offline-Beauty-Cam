import numpy as np
import math


# 高斯分布
def gaussian_distribution(x, y, sigma):
    return pow(math.e, - (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))) / (2 * math.pi * pow(sigma, 2))


# 生成高斯卷积核，radius为卷积核半径
def create_gaussian_kernel(sigma, radius):
    k_shape = (radius * 2 - 1, radius * 2 - 1)
    k_center = (radius - 1, radius - 1)
    kernel = np.zeros(k_shape, float)
    for i in range(0, k_shape[0]):
        for j in range(0, k_shape[1]):
            kernel[i, j] = gaussian_distribution(abs(k_center[0] - i), abs(k_center[1] - j), sigma)
    k_sum = kernel.sum()
    for i in range(0, k_shape[0]):
        for j in range(0, k_shape[1]):
            kernel[i, j] /= k_sum
    return kernel


# 图片外围填充
def pad_image(image: np.ndarray, radius):
    pad_width = radius - 1
    for i in range(pad_width):
        image = np.hstack((image[:, :1], image, image[:, -1:]))
    for j in range(pad_width):
        image = np.vstack((image[:1, :], image, image[-1:, :]))
    return image


# 卷积
def convolution(image: np.ndarray, kernel: np.ndarray):
    k_width = kernel.shape[0]
    pad_width = int((k_width - 1) / 2)
    shape = (int(image.shape[0] - 2 * pad_width), int(image.shape[1] - 2 * pad_width))
    new_image = np.zeros(shape, int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_image[i, j] = (image[i: i + k_width, j: j + k_width] * kernel).sum()
    return new_image


# 高斯模糊
def gaussian_blur(image: np.ndarray, sigma, radius):
    kernel = create_gaussian_kernel(sigma, radius)
    image_b = image[:, :, 0:1].reshape(image.shape[:2])
    image_g = image[:, :, 1:2].reshape(image.shape[:2])
    image_r = image[:, :, 2:3].reshape(image.shape[:2])
    padded_image_b = pad_image(image_b, radius)
    padded_image_g = pad_image(image_g, radius)
    padded_image_r = pad_image(image_r, radius)
    new_image_b = convolution(padded_image_b, kernel)
    new_image_g = convolution(padded_image_g, kernel)
    new_image_r = convolution(padded_image_r, kernel)
    new_image = np.dstack((new_image_b, new_image_g, new_image_r))
    return new_image
