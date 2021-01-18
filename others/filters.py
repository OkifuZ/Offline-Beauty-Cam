import cv2
import numpy as np
import math


class Filters:

    def __init__(self, src: np.ndarray):
        self.cheat_sheet = np.zeros(256, float)
        self.src_color = src
        self.src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        self.space_kernel = None
        self.color_kernel = None
        self.real_kernel = None
        self.shape_1 = src.shape[0]
        self.shape_2 = src.shape[1]
        self.radius = -1

    @staticmethod
    def pad_image(src, radius):
        pad_width = radius - 1
        constant = cv2.copyMakeBorder(src, pad_width, 10, 10, 10, cv2.BORDER_REPLICATE)
        return constant

    @staticmethod
    def gaussian_distribution_2d(x, y, sigma):
        return pow(math.e, - (pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))) / (2 * math.pi * pow(sigma, 2))

    @staticmethod
    def gaussian_distribution_1d(x, sigma):
        return pow(math.e, - pow(x, 2)) / (2 * pow(sigma, 2)) / math.sqrt(2 * math.pi * pow(sigma, 2))

    def create_space_kernel(self, sigma):
        k_shape = (self.radius * 2 - 1, self.radius * 2 - 1)
        k_center = (self.radius - 1, self.radius - 1)
        kernel = np.zeros(k_shape, float)
        for i in range(0, k_shape[0]):
            for j in range(0, k_shape[1]):
                kernel[i, j] = self.gaussian_distribution_2d(abs(k_center[0] - i), abs(k_center[1] - j), sigma)
        k_sum = kernel.sum()
        for i in range(0, k_shape[0]):
            for j in range(0, k_shape[1]):
                kernel[i, j] /= k_sum
        return kernel

    def create_color_kernel(self, i, j, k_width, pad_width):
        for i_ in range(k_width):
            for j_ in range(k_width):
                diff_gray = \
                    int(abs(int(self.src_gray[i + i_ - pad_width][j + j_ - pad_width]) - int(self.src_gray[i][j])))
                self.color_kernel[i_][j_] = self.cheat_sheet[diff_gray]

    def bi_convolution(self, src_channel: np.ndarray) -> np.ndarray:
        print('finished one channel')
        k_width = self.radius * 2 - 1
        pad_width = self.radius - 1
        new_image = np.zeros((self.shape_1, self.shape_2), int)
        for i in range(self.shape_1):
            for j in range(self.shape_2):
                self.color_kernel = np.zeros((k_width, k_width), float)
                self.create_color_kernel(i, j, k_width, pad_width)
                self.real_kernel = self.color_kernel * self.space_kernel
                k_sum = self.real_kernel.sum()
                for i_ in range(0, k_width):
                    for j_ in range(0, k_width):
                        self.real_kernel[i_, j_] /= k_sum
                new_image[i, j] = (src_channel[i: i + k_width, j: j + k_width] * self.real_kernel).sum()
        return new_image

    def bilateral_filter(self, sigma_color, sigma_space, radius) -> np.ndarray:
        self.radius = radius
        self.space_kernel = self.create_space_kernel(sigma_space)
        for i in range(256):
            self.cheat_sheet[i] = self.gaussian_distribution_1d(float(i), sigma_color)
        # pad gray img
        self.src_gray = self.pad_image(self.src_gray, self.radius)
        # pad color img
        src_b = self.src_color[:, :, 0:1].reshape(self.src_color.shape[:2])
        src_g = self.src_color[:, :, 1:2].reshape(self.src_color.shape[:2])
        src_r = self.src_color[:, :, 2:3].reshape(self.src_color.shape[:2])
        src_b = self.pad_image(src_b, self.radius)
        src_g = self.pad_image(src_g, self.radius)
        src_r = self.pad_image(src_r, self.radius)
        # conv
        new_image_b = self.bi_convolution(src_b)
        new_image_g = self.bi_convolution(src_g)
        new_image_r = self.bi_convolution(src_r)

        new_image = np.dstack((new_image_b, new_image_g, new_image_r))
        return new_image

