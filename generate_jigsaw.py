import cv2
import numpy as np

def generate_jigsaw_img(img):
    # cv2.imshow('my_img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    h, w = img.shape[:2]
    print(f"width {w} height {h}")
    
    h_clean = h - (np.mod(h, 4))
    w_clean = w - (np.mod(w, 4))
    block_hw = (h_clean // 4, w_clean // 4)
    
def get_bezier_point(t, p0, p1, p2, p3):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    x_new = (1-t)**3 * x0 + 3*(1-t)**2 * t*x1 + 3*(1-t) * t**2 * x2 + t**3 * x3
    y_new = (1-t)**3 * y0 + 3*(1-t)**2 * t*y1 + 3*(1-t) * t**2 * y2 + t**3 * y3
    return x_new, y_new

def generate_curve(num_points, p0, p1, p2, p3):
    path = []
    
    for i in range(num_points):
        t = i / (num_points - 1)
        point = get_bezier_point(t, p0, p1, p2, p3)
        path.append(point)
    
    return path

if __name__ == "__main__":
    image = cv2.imread('img/Grumpy_Kitty.png')
    generate_jigsaw_img(image)