import cv2
import filters

if __name__ == '__main__':
    src = cv2.imread('../RAWPIC/3.jpg', cv2.IMREAD_COLOR)
    my_filter = filters.Filters(src)

    show = my_filter.bilateral_filter(10.0, 4.0, 4)
    cv2.imwrite('ans.jpg', show)
