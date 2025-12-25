import cv2
import numpy as np
import os
from generate_jigsaw import generate_jigsaw_img
from reconstruct_from_csv import reconstruct_puzzle

# Configuration
TEST_IMG = "img/Grumpy_Kitty.png"
PUZZLE_DIMS = 10

def test_round_trip_reconstruction():
    """
    Verifies that Image -> Jigsaw -> Reconstruction results in 
    a nearly identical image to the source.
    """  
    # Run Generation (Creates CSV + Spritesheet)
    generate_jigsaw_img(TEST_IMG, PUZZLE_DIMS)
    
    base_name = os.path.splitext(os.path.basename(TEST_IMG))[0]
    csv_path = f"jigsaw/{base_name}_data.csv"
    sheet_path = f"jigsaw/{base_name}_spritesheet.png"
    
    assert os.path.exists(csv_path), "CSV not generated"
    assert os.path.exists(sheet_path), "Spritesheet not generated"

    # Run Reconstruction (Get the image back as array)
    reconstruct_puzzle(csv_path, sheet_path)
    reconstructed_path = f"reconstructed/{base_name}_reconstructed.png"
    
    reconstructed_img = cv2.imread(reconstructed_path)
    original_img = cv2.imread(TEST_IMG)
    
    assert reconstructed_img is not None, f"Could not find reconstructed image at: {reconstructed_path}"
    assert original_img is not None, "Could not load original test image."

    # Handle Dimension Mismatch (Modulo Cropping)
    # The generator trims the original image to fit the grid.
    rh, rw = reconstructed_img.shape[:2]
    oh, ow = original_img.shape[:2]
    
    # Crop original to match the 'clean' dimensions used by generator
    min_h = min(rh, oh)
    min_w = min(rw, ow)
    
    original_crop = original_img[0:min_h, 0:min_w]
    reconstructed_crop = reconstructed_img[0:min_h, 0:min_w]

    # The Subtraction Test
    # We use absdiff to prevent uint8 overflow
    diff = cv2.absdiff(original_crop, reconstructed_crop)
    
    # Check for Success
    # Allow a tiny tolerance because Alpha Blending (float math) -> Uint8 conversion
    max_diff = np.max(diff)
    
    # We assert that the maximum pixel difference is very low (e.g., < 2)
    assert max_diff <= 2, f"Reconstruction failed! Max pixel deviation: {max_diff}"