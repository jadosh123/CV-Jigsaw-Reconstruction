import cv2
import numpy as np
import random
import csv
import os
import argparse

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


def generate_jigsaw_img(img_path, puzzle_dims: int):
    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not decode image. Check file format.")
        return
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = "jigsaw"
    os.makedirs(output_dir, exist_ok=True)
    
    sheet_name = os.path.join(output_dir, f"{base_name}_spritesheet.png")
    csv_name = os.path.join(output_dir, f"{base_name}_data.csv")
        
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

    sheet = create_jigsaw_spritesheet(img, jigsaw_mat, bh, bw, csv_name)
    cv2.imwrite(sheet_name, sheet)
    print(f"Saved: {sheet_name}")
            
def extract_piece(img, contour):
    """
    Extracts a piece safely, handling cases where the contour 
    might slightly exceed image boundaries (clipping).
    """
    x, y, w, h = cv2.boundingRect(contour)
    ih, iw = img.shape[:2]
    
    src_x1 = max(0, x)
    src_y1 = max(0, y)
    src_x2 = min(iw, x + w)
    src_y2 = min(ih, y + h)
    
    # Check if we have any valid overlap
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return np.zeros((h, w, 4), dtype=np.uint8)

    valid_crop = img[src_y1:src_y2, src_x1:src_x2]
    roi_full = np.zeros((h, w, 3), dtype=np.uint8)
    
    dst_x = src_x1 - x
    dst_y = src_y1 - y
    
    crop_h, crop_w = valid_crop.shape[:2]
    
    # Paste the piece at the proper coordinates 
    # inside the bigger roi map
    roi_full[dst_y : dst_y + crop_h, dst_x : dst_x + crop_w] = valid_crop
    
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = contour - np.array([x, y])
    cv2.fillPoly(mask, [shifted_contour], 255)
    
    b, g, r = cv2.split(roi_full)
    rgba = cv2.merge([b, g, r, mask])
    
    return rgba

def create_jigsaw_spritesheet(img, jigsaw_mat, bh, bw, output_csv="jigsaw/puzzle_data.csv"):
    """
    Creates a transparent spritesheet of all pieces.
    """
    rows = len(jigsaw_mat)
    cols = len(jigsaw_mat[0])
    
    max_w = 0
    max_h = 0
    all_contours = [] 

    for row in range(rows):
        # row_contours = []
        for col in range(cols):
            piece = jigsaw_mat[row][col]
            if piece is None: continue
            
            contour_points = []
            contour_points.extend(piece.edges["TOP"][:-1])
            contour_points.extend(piece.edges["RIGHT"][:-1])
            contour_points.extend(piece.edges["BOTTOM"][:-1])
            contour_points.extend(piece.edges["LEFT"][:-1])
            contour_np = np.array(contour_points, dtype=np.int32)
            
            # row_contours.append(contour_np)
            
            x, y, w, h = cv2.boundingRect(contour_np)
            if w > max_w: max_w = w
            if h > max_h: max_h = h
            
            all_contours.append({
                "grid_row": row,
                "grid_col": col,
                "contour": contour_np,
                "piece_obj": piece
            })

        # all_contours.append(row_contours)

    padding = 2
    cell_w = max_w + padding
    cell_h = max_h + padding

    sheet_cols = cols
    sheet_rows = rows
    sheet_w = cell_w * cols
    sheet_h = cell_h * rows
    
    print(f"Generating Spritesheet: {sheet_w}x{sheet_h} pixels")
    spritesheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    
    random.shuffle(all_contours)
    
    data = []
    
    for slot_idx, item in enumerate(all_contours):
        slot_row = slot_idx // sheet_cols
        slot_col = slot_idx % sheet_cols
        
        target_x = slot_col * cell_w
        target_y = slot_row * cell_h
        
        # Extract the individual piece
        contour = item["contour"]
        piece_img = extract_piece(img, contour)
        ph, pw = piece_img.shape[:2]
        
        spritesheet[target_y : target_y + ph, target_x : target_x + pw] = piece_img
        cnt_x, cnt_y, cnt_w, cnt_h = cv2.boundingRect(contour)
        
        data.append({
            "id": slot_idx,
            "grid_row": item["grid_row"],
            "grid_col": item["grid_col"],
            "orig_x": cnt_x,
            "orig_y": cnt_y,
            "sheet_x": target_x,
            "sheet_y": target_y,
            "edge_top": item["piece_obj"].top,
            "edge_right": item["piece_obj"].right,
            "edge_bottom": item["piece_obj"].bottom,
            "edge_left": item["piece_obj"].left,
            "block_height": bh,
            "block_width": bw,
            "sprite_width": pw,
            "sprite_height": ph
        })

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        print(f"Saved metadata to {output_csv}")

    return spritesheet

def generate_jigsaw_pieces(num_pcs_y, num_pcs_x):
    jigsaw_mat = [[Jigsaw_Piece(None, None, None, None) for _ in range(num_pcs_x)] for _ in range(num_pcs_y)]

    for row in range(num_pcs_y):
        for col in range(num_pcs_x):
            curr_pc = jigsaw_mat[row][col]
            
            # Top
            if row == 0: top = "FLAT"
            elif jigsaw_mat[row-1][col].bottom == "TAB": top = "SLOT"
            else: top = "TAB"
            curr_pc.top = top

            # Right
            if col == num_pcs_x - 1: right = "FLAT"
            else: right = PIECE_OPT[random.randint(0, 1)]
            curr_pc.right = right
            
            # Bottom
            if row == num_pcs_y - 1: bottom = "FLAT"
            else: bottom = PIECE_OPT[random.randint(0, 1)]
            curr_pc.bottom = bottom
            
            # Left
            if col == 0: left = "FLAT"
            elif jigsaw_mat[row][col-1].right == "TAB": left = "SLOT"
            else: left = "TAB"
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
    
    points = []
    if edge_type == "TAB":
        for segment in TAB_PROFILE:
            segment_points = generate_curve(20, *segment)
            points.extend(segment_points)
    elif edge_type == "SLOT": 
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
        
        if side == "TOP":
            rx, ry = sx, sy
        elif side == "RIGHT":
            rx, ry = -sy, sx
        elif side == "BOTTOM":
            rx, ry = -sx, -sy
        elif side == "LEFT":
            rx, ry = sy, -sx

        new_points.append((int(off_x + rx), int(off_y + ry)))
    
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
    parser = argparse.ArgumentParser(description="Jigsaw Puzzle Generator")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--dims", type=int, default=10, help="Grid size (e.g., 10 for 10x10 puzzle)")
    
    args = parser.parse_args()
    
    generate_jigsaw_img(args.image_path, args.dims)