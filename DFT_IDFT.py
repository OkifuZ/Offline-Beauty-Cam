import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

name = input('picture name: ')
print('if you need idft, please replace pos with your light pots position list in the code!')
need_idft = (input('need idft? :(y/n): ') == 'y')


# 添加规律噪声
img = cv.imread(name, cv.IMREAD_COLOR)
m = img.shape[0]
n = img.shape[1]
cv.imwrite('../../processed/4_raw_color.jpg', img)
for k in range(3):
    for i in range(m):
        for j in range(n):
            new_pixel = img[i][j][k] + 20*math.sin(20*i)+20*math.sin(20*j)
            if new_pixel > 255:
                new_pixel = 255
            elif new_pixel < 0:
                new_pixel = 0
            img[i][j][k] = new_pixel

f = np.fft.fft2(img)
f_shift = np.fft.fftshift(f)
ms = 20*np.log(np.abs(f_shift))

# 展示频谱图
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(ms, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])

plt.savefig('mag_'+name)
plt.show()

if need_idft:
    radius = 10
    pos = list()
    '''
    replace pos with your light pots' position list 
    '''

    # 去除亮斑，置为0
    for i in range(len(pos)):
        for k1 in range(radius):
            for k2 in range(radius):
                f_shift[k1+pos[0]-radius//2][k2+pos[1]-radius//2] = 0 # 去除亮斑

    f_processed = np.fft.ifftshift(f_shift)
    img_processed = np.fft.ifft2(f_processed) # 回复原图像
    cv.imshow('idft_'+name, img_processed)


