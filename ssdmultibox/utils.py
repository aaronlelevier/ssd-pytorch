import cv2


def open_image(image_path):
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    return cv2.imread(image_path, flags)/255
