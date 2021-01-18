import cv2
import numpy as np
import math


class anisodiff2D(object):

    def __init__(self, num_iter=5, delta_t=1 / 7, kappa=30, option=2):

        super(anisodiff2D, self).__init__()

        self.num_iter = num_iter
        self.delta_t = delta_t
        self.kappa = kappa
        self.option = option

        self.hN = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        self.hS = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
        self.hE = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        self.hW = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
        self.hNE = np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]])
        self.hSE = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.hSW = np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]])
        self.hNW = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

    def fit(self, img):

        diff_im = img.copy()

        dx = 1
        dy = 1
        dd = math.sqrt(2)

        for i in range(self.num_iter):
            nablaN = cv2.filter2D(diff_im, -1, self.hN)
            nablaS = cv2.filter2D(diff_im, -1, self.hS)
            nablaW = cv2.filter2D(diff_im, -1, self.hW)
            nablaE = cv2.filter2D(diff_im, -1, self.hE)
            nablaNE = cv2.filter2D(diff_im, -1, self.hNE)
            nablaSE = cv2.filter2D(diff_im, -1, self.hSE)
            nablaSW = cv2.filter2D(diff_im, -1, self.hSW)
            nablaNW = cv2.filter2D(diff_im, -1, self.hNW)

            cN = 0
            cS = 0
            cW = 0
            cE = 0
            cNE = 0
            cSE = 0
            cSW = 0
            cNW = 0

            if self.option == 1:
                cN = np.exp(-(nablaN / self.kappa) ** 2)
                cS = np.exp(-(nablaS / self.kappa) ** 2)
                cW = np.exp(-(nablaW / self.kappa) ** 2)
                cE = np.exp(-(nablaE / self.kappa) ** 2)
                cNE = np.exp(-(nablaNE / self.kappa) ** 2)
                cSE = np.exp(-(nablaSE / self.kappa) ** 2)
                cSW = np.exp(-(nablaSW / self.kappa) ** 2)
                cNW = np.exp(-(nablaNW / self.kappa) ** 2)
            elif self.option == 2:
                cN = 1 / (1 + (nablaN / self.kappa) ** 2)
                cS = 1 / (1 + (nablaS / self.kappa) ** 2)
                cW = 1 / (1 + (nablaW / self.kappa) ** 2)
                cE = 1 / (1 + (nablaE / self.kappa) ** 2)
                cNE = 1 / (1 + (nablaNE / self.kappa) ** 2)
                cSE = 1 / (1 + (nablaSE / self.kappa) ** 2)
                cSW = 1 / (1 + (nablaSW / self.kappa) ** 2)
                cNW = 1 / (1 + (nablaNW / self.kappa) ** 2)

            diff_im = diff_im + self.delta_t * (

                    (1 / dy ** 2) * cN * nablaN +
                    (1 / dy ** 2) * cS * nablaS +
                    (1 / dx ** 2) * cW * nablaW +
                    (1 / dx ** 2) * cE * nablaE +

                    (1 / dd ** 2) * cNE * nablaNE +
                    (1 / dd ** 2) * cSE * nablaSE +
                    (1 / dd ** 2) * cSW * nablaSW +
                    (1 / dd ** 2) * cNW * nablaNW
            )

        return diff_im

if __name__ == '__main__':
    src = cv2.imread('../RAWPIC/5.jpg', cv2.IMREAD_COLOR)
    obj = anisodiff2D()
    show = obj.fit(src)
    cv2.imwrite('ani.jpg', show)