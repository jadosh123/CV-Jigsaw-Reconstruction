import cv2
import numpy as np
import random
from typing import List, Tuple, Optional

TAB_PROFILE = [
    ((0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0)),
    ((0.3, 0.0), (0.2, -0.25), (0.8, -0.25), (0.7, 0.0)),
    ((0.7, 0.0), (0.8, 0.0), (0.9, 0.0), (1.0, 0.0))
]

PIECE_OPT = ["SLOT", "TAB"]

class Jigsaw_Piece():
    def __init__(self, top, right, bottom, left):
        self.top: str = top
        self.right: str = right
        self.bottom: str = bottom
        self.left: str = left
        self.edges = {"TOP": [], "RIGHT": [], "BOTTOM": [], "LEFT": []}


def generate_jigsaw_img(img, puzzle_dims: int):
    h, w = img.shape[:2]
    print(f"width {w} height {h}")
    
    if puzzle_dims is None:
        print("Please provide the puzzle dimensions (height, width).")
        return
    
    num_pcs_y, num_pcs_x = puzzle_dims, puzzle_dims
    h_clean = h - (np.mod(h, num_pcs_y))
    w_clean = w - (np.mod(w, num_pcs_x))
    block_hw = (h_clean // num_pcs_y, w_clean // num_pcs_x)
    bh, bw = block_hw[0], block_hw[1]
    
    jigsaw_mat = generate_jigsaw_pieces(num_pcs_y, num_pcs_x)
    
    for row in range(num_pcs_y):
        for col in range(num_pcs_x):
            tl = (col * bw, row * bh)
            tr = ((col + 1) * bw, row * bh)
            br = ((col + 1) * bw, (row + 1) * bh)
            bl = (col * bw, (row + 1) * bh)
            curr_jig = jigsaw_mat[row][col]
            
            if curr_jig is None:
                print(f"Encountered None piece at [{row},{col}].")
                return
            
            curr_jig.edges["TOP"] = create_piece_edge(tl, tr, curr_jig.top, "TOP", bw)
            curr_jig.edges["RIGHT"] = create_piece_edge(tr, br, curr_jig.right, "RIGHT", bh)
            curr_jig.edges["BOTTOM"] = create_piece_edge(br, bl, curr_jig.bottom, "BOTTOM", bw)
            curr_jig.edges["LEFT"] = create_piece_edge(bl, tl, curr_jig.left, "LEFT", bh)

            contour = []
            contour.extend(curr_jig.edges["TOP"][:-1])
            contour.extend(curr_jig.edges["RIGHT"][:-1])
            contour.extend(curr_jig.edges["BOTTOM"][:-1])
            contour.extend(curr_jig.edges["LEFT"][:-1])

            contour_np = np.array(contour, dtype=np.int32)
            cv2.polylines(img, [contour_np], isClosed=True, color=(255, 0, 0), thickness=10)
    
    # 4. Show the result
    cv2.imshow('Jigsaw Preview', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
            

def generate_jigsaw_pieces(num_pcs_y, num_pcs_x):
    jigsaw_mat = [[Jigsaw_Piece(None, None, None, None) for _ in range(num_pcs_x)] for _ in range(num_pcs_y)]

    for row in range(num_pcs_y):
        for col in range(num_pcs_x):
            curr_pc = jigsaw_mat[row][col]
            
            if row == 0:
                top = "FLAT"
            elif jigsaw_mat[row-1][col].bottom == "TAB":
                top = "SLOT"
            else:
                top = "TAB"
            curr_pc.top = top

            if col == num_pcs_x - 1:
                right = "FLAT"
            else:
                right = PIECE_OPT[random.randint(0, 1)]
            curr_pc.right = right
            
            if row == num_pcs_y - 1:
                bottom = "FLAT"
            else:
                bottom = PIECE_OPT[random.randint(0, 1)]
            curr_pc.bottom = bottom
            
            if col == 0:
                left = "FLAT"
            elif jigsaw_mat[row][col-1].right == "TAB":
                left = "SLOT"
            else:
                left = "TAB"
            curr_pc.left = left
    
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
    
    print(f"Warning: Unknown edge type '{edge_type}'")
    return []
        
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
        
        if side == "TOP":
            rx, ry = sx, sy
        elif side == "RIGHT":
            rx, ry = -sy, sx
        elif side == "BOTTOM":
            rx, ry = -sx, -sy
        elif side == "LEFT":
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
    generate_jigsaw_img(image, 10)