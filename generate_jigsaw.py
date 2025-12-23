import cv2
import numpy as np
import random
from typing import List, Tuple, Optional

TAB_PROFILE = [
    ((0.0, 0.0), (0.2, 0.0), (0.3, -0.15), (0.35, -0.15)),
    ((0.35, -0.15), (0.2, -0.5), (0.8, -0.5), (0.65, -0.15)),
    ((0.65, -0.15), (0.7, -0.15), (0.8, 0.0), (1.0, 0.0))
]

PIECE_OPT = ["SLOT", "TAB"]

class Jigsaw_Piece():
    def __init__(self, top, right, bottom, left):
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left
        

def generate_jigsaw_img(img, puzzle_dims: int):
    # cv2.imshow('my_img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    h, w = img.shape[:2]
    print(f"width {w} height {h}")
    
    if puzzle_dims is None:
        print("Please provide the puzzle dimensions (height, width).")
        return
    
    num_pcs_y, num_pcs_x = puzzle_dims, puzzle_dims
    h_clean = h - (np.mod(h, num_pcs_y))
    w_clean = w - (np.mod(w, num_pcs_x))
    block_hw = (h_clean // num_pcs_y, w_clean // num_pcs_x)
    
    jigsaw_mat = generate_jigsaw_pieces(num_pcs_y, num_pcs_x)

def generate_jigsaw_pieces(num_pcs_y, num_pcs_x):
    jigsaw_mat: List[List[Optional[Jigsaw_Piece]]] = [[None for _ in range(num_pcs_x)] for _ in range(num_pcs_y)]

    for row in range(num_pcs_y):
        for col in range(num_pcs_x):
            if row == 0:
                top = "FLAT"
            elif jigsaw_mat[row-1][col].bottom == "TAB":
                top = "SLOT"
            else:
                top = "TAB"

            if col == num_pcs_x - 1:
                right = "FLAT"
            else:
                right = PIECE_OPT[random.randint(0, 1)]
            
            if row == num_pcs_y - 1:
                bottom = "FLAT"
            else:
                bottom = PIECE_OPT[random.randint(0, 1)]
            
            if col == 0:
                left = "FLAT"
            elif jigsaw_mat[row][col-1].right == "TAB":
                left = "SLOT"
            else:
                left = "TAB"

            piece = Jigsaw_Piece(top, right, bottom, left)
            jigsaw_mat[row][col] = piece
    
    return jigsaw_mat

def create_piece_edge(start, end, edge_type, side, block_size):
    """
    Creates a Flat/Tab/Slot edge piece
    
    :param start: first edge corner
    :param end: last edge corner
    :param edge_type: FLAT or TAB or SLOT
    :param side: The side the edge is on (top, right, bottom, left)
    :param block_size: Size of piece
    """
    if edge_type == "FLAT":
        return [start, end]
    
    if edge_type == "TAB":
        points = []
        for segment in TAB_PROFILE:
            segment_points = generate_curve(20, *segment)
            points.extend(segment_points)

        return transform_points(points, side, block_size, start)
    elif edge_type == "SLOT":
        points = []
        
        for segment in TAB_PROFILE:
            inverted_segment = [(num[0], -num[1]) for num in segment]    
            segment_points = generate_curve(20, *inverted_segment)
            points.extend(segment_points)
            
        return transform_points(points, side, block_size, start)
        
def transform_points(points, side, block_size, offset):
    """
    Transform point to reflect real coordinates on image
    
    :param points: A list of the raw normalized tuples from TAB_PROFILE
    :param side: A string ("top", "bottom", "left", "right")
    :param block_size: Integer
    :param offset: Tuple(x, y) for the starting corner
    """
    new_points = []
    off_x, off_y = offset
    
    for x, y in points:
        sx = x * block_size
        sy = y * block_size
        
        if side == "top":
            rx, ry = sx, sy
        elif side == "right":
            rx, ry = -sy, sx
        elif side == "bottom":
            rx, ry = -sx, -sy
        elif side == "left":
            rx, ry = sy, -sx
        else:
            print("Invalid side provided use (top, right, bottom, left)")
            return []
    
        final_x = off_x + rx
        final_y = off_y + ry
        new_points.append((int(final_x), int(final_y)))
    
    return new_points

def get_bezier_point(t, p0, p1, p2, p3):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    x_new = (1-t)**3 * x0 + 3*(1-t)**2 * t*x1 + 3*(1-t) * t**2 * x2 + t**3 * x3
    y_new = (1-t)**3 * y0 + 3*(1-t)**2 * t*y1 + 3*(1-t) * t**2 * y2 + t**3 * y3
    return (x_new, y_new)

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