import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./4_mb.png', cv.IMREAD_GRAYSCALE)
f = np.fft.fft2(img)
f_shift = np.fft.fftshift(f)
ms = 20*np.log(np.abs(f_shift))


plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(ms, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([]), plt.yticks([])

plt.savefig('mb_ff_4.jpg')
plt.show()
