import cv2
import math
import numpy as np


img = cv2.imread('../../RAWPIC/4.jpg', cv2.IMREAD_COLOR)
m = img.shape[0]
n = img.shape[1]
cv2.imwrite('../../processed/4_raw_color.jpg', img)
for k in range(3):
    for i in range(m):
        for j in range(n):
            new_pixel = img[i][j][k] + 20*math.sin(20*i)+20*math.sin(20*j)
            if new_pixel > 255:
                new_pixel = 255
            elif new_pixel < 0:
                new_pixel = 0
            img[i][j][k] = new_pixel
cv2.imwrite('../../processed/4_with_reg_noise_color.jpg', img)
cv2.waitKey(0)