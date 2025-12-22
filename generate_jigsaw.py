import cv2
import numpy as np

def generate_jigsaw_img(img):
    cv2.imshow(img)

if __name__ == "__main__":
    image = cv2.imread('img/Grumpy_Kitty.png')
    generate_jigsaw_img(image)