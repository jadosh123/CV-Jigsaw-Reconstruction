import cv2
import numpy as np
import csv
import argparse
import os

def reconstruct_puzzle(csv_path, sheet_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return
    if not os.path.exists(sheet_path):
        print(f"Error: Spritesheet not found at {sheet_path}")
        return

    print(f"Loading spritesheet: {sheet_path}")
    spritesheet = cv2.imread(sheet_path, cv2.IMREAD_UNCHANGED)
    if spritesheet is None:
        print("Error: Failed to load image.")
        return

    pieces = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pieces.append(row)
    
    print(f"Loaded {len(pieces)} pieces from metadata.")

    max_x = 0
    max_y = 0
    
    for p in pieces:
        right_edge = int(p["orig_x"]) + int(p["sprite_width"])
        bottom_edge = int(p["orig_y"]) + int(p["sprite_height"])
        
        if right_edge > max_x: max_x = right_edge
        if bottom_edge > max_y: max_y = bottom_edge

    print(f"Reconstructed Canvas Size: {max_x}x{max_y}")
    
    canvas = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    for i, p in enumerate(pieces):
        sx = int(p["sheet_x"])
        sy = int(p["sheet_y"])
        w = int(p["sprite_width"])
        h = int(p["sprite_height"])
        
        dx = int(p["orig_x"])
        dy = int(p["orig_y"])
        
        sprite = spritesheet[sy:sy+h, sx:sx+w]
        
        # Normalize alpha to 0.0 - 1.0
        alpha = sprite[:, :, 3] / 255.0
        # Get RGB channels
        piece_rgb = sprite[:, :, :3]
        
        # Get the corresponding region on the canvas
        canvas_roi = canvas[dy:dy+h, dx:dx+w]
        
        # Blend: (NewPixel * Alpha) + (OldPixel * (1 - Alpha))
        # We need to broadcast alpha dimensions to match RGB (h, w, 1)
        alpha_factor = alpha[:, :, np.newaxis]
        
        blended = (piece_rgb * alpha_factor) + (canvas_roi * (1.0 - alpha_factor))
        
        # Write back to canvas
        canvas[dy:dy+h, dx:dx+w] = blended.astype(np.uint8)

    output_dir = "reconstructed"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(csv_path).replace("_data.csv", "")
    output_filename = os.path.join(output_dir, f"{base_name}_reconstructed.png")    

    cv2.imwrite(output_filename, canvas)
    print(f"Success! Reconstructed image saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct Jigsaw from CSV")
    parser.add_argument("--csv", required=True, help="Path to puzzle_data.csv")
    parser.add_argument("--sheet", required=True, help="Path to spritesheet.png")
    
    args = parser.parse_args()
    
    reconstruct_puzzle(args.csv, args.sheet)