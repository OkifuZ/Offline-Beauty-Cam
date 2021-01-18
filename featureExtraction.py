import cv2
import dlib
import numpy as np
import os


def extract_feature(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    predictor = dlib.shape_predictor("./MODEL/shape_predictor_68_face_landmarks.dat")
    ans_x = list()
    ans_y = list()
    for det, d in enumerate(dets):
        print(det)
        arr_x = np.zeros(68)
        arr_y = np.zeros(68)
        shape = predictor(img, d)
        for i in range(68):
            cv2.line(img, (shape.part(i).x, shape.part(i).y), (shape.part(i).x, shape.part(i).y), (0, 255, 0))
            arr_x[i] = shape.part(i).x
            arr_y[i] = shape.part(i).y
        np.save('feature_x.npy', arr_x)
        np.save('feature_y.npy', arr_y)
        ans_x.append(arr_x)
        ans_y.append(arr_y)
    return ans_x, ans_y


