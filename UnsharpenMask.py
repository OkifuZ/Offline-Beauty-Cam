import cv2


if __name__ == '__main__':
    name = input('picture name: ')
    time = input('sharpen power:(int) ')
    src = cv2.imread(name, cv2.IMREAD_COLOR)
    for i in range(int(time)):
        low_freq = cv2.GaussianBlur(src, (9, 9), 1.0)
        template = cv2.subtract(src, low_freq)
        src = cv2.add(src, template)
    cv2.imwrite('sharpen_'+name, src)
