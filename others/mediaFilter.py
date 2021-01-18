import cv2
import numpy as np
# Median filter


def median_filter(img, K_size=3):
    H, W, C = img.shape
    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad:pad + H, pad:pad + W] = img.copy().astype(np.float)
    tmp = out.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.median(tmp[y:y + K_size, x:x + K_size, c])
    out = out[pad:pad + H, pad:pad + W].astype(np.uint8)
    return out
# Average filter


def average_filter(img, G=3):
    out = img.copy()
    H, W, C = img.shape
    Nh = int(H / G)
    Nw = int(W / G)
    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[G * y:G * (y + 1), G * x:G * (x + 1), c] = np.mean(
                    out[G * y:G * (y + 1), G * x:G * (x + 1), c]).astype(np.int)
    return out


# Read image
img = cv2.imread("../report/5_pepper.jpg")
# Median Filter and Average Filter
out1 = median_filter(img, K_size=3)
out2 = average_filter(img, G=3)
# Save result
cv2.imwrite("../report/media_out1.jpg", out1)
cv2.imwrite("../report/media_out2.jpg", out2)
cv2.waitKey(0)
cv2.destroyAllWindows()
