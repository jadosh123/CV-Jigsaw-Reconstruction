# Computer Vision Jigsaw Reconstruction
A Python-based framework for generating, playing, and conceptually solving jigsaw puzzles using Computer Vision. This project bridges the gap between synthetic data generation and interactive gameplay, serving as a robust testbed for future AI solver implementations (SIFT/RANSAC, Deep Learning).

## Features
1. Procedural Puzzle Generation (`generate_jigsaw.py`)
    - **Intelligent Cutting:** Automatically segments any image into a jigsaw grid with varying edge types (Tab, Slot, Flat).
    - **Bezier Curve Geometry:** Uses mathematical Bezier curves to create realistic, smooth puzzle piece contours.
    - **Randomized Shuffling:** Pieces are randomly distributed on the spritesheet to prevent trivial reconstruction based on file order, ensuring solvers must rely on visual features or coordinate logic.
    - **Metadata Manifest:** Exports a detailed CSV (`puzzle_data.csv`) containing ground truth data for every piece (coordinates, edge types, dimensions), essential for AI training and validation.

2. Reconstruction & Validation (`reconstruct_from_csv.py`)
    - **Lossless Reconstruction:** Mathematically reassembles the puzzle using the generated CSV data to prove dataset integrity.
    - **Alpha Blending:** Handles transparency and overlapping tabs/slots correctly to reconstruct the original image without artifacts.

## Project Structure
```
CV-Jigsaw-Reconstruction/
├── example_output/
│   ├── Grumpy_Kitty_data.csv
│   └── Grumpy_Kitty_spritesheet.png
├── img/
│   └── Grumpy_Kitty.png       # Source image for the puzzle
├── jigsaw/
│   ├── jigsaw_spritesheet.png # Generated transparent spritesheet
│   └── puzzle_data.csv        # Generated ground truth manifest
├── generate_jigsaw.py         # Script to cut image & create dataset
├── requirements.txt           # Dependencies
├── reconstruct_from_csv.py    # Reconstructs image using csv and sprite
├── test_jigsaw.py             # Tests reconstruction output
└── README.md
```

## Getting Started
**Prerequisites**
- Python 3.7+
- OpenCV (`opencv-python`)
- Numpy
- Pytest (optional, for testing)

**Run in the terminal**:
```bash
pip install opencv-python numpy pytest
```

## 1. Generating a Puzzle
Run the generator to create a new puzzle. The output will be saved to the jigsaw/ folder.

```bash
python generate_jigsaw.py img/Grumpy_Kitty.png
```

**Custom Grid Size:** You can specify the puzzle dimensions (e.g., 20x20) using the --dims flag.

```bash
python generate_jigsaw.py img/Grumpy_Kitty.png --dims 20
```
- **Input:** Path to any image file.
- **Output:** Creates jigsaw/(filename)_spritesheet.png and jigsaw/(filename)_data.csv

## 2. Reconstructing (Validation)
To verify that the generated data is correct, you can run the reconstruction script. This reads the shuffled CSV instructions and rebuilds the original image.

```bash
python reconstruct_from_csv.py --csv jigsaw/Grumpy_Kitty_data.csv --sheet jigsaw/Grumpy_Kitty_spritesheet.png
```

## 3. Automated Testing
To mathematically prove that the reconstruction matches the original source image (pixel-perfect integrity), run the test suite:
```bash
python -m pytest test_jigsaw.py
```
OR run this in root directory.
```bash
pytest
```

## Dataset Structure (for AI)
The puzzle_data.csv is designed for training Computer Vision models. It contains the following fields:

| Field | Description |
| :--- | :--- |
| `id` | Unique identifier for the piece. |
| `grid_row`, `grid_col` | Logical position in the puzzle grid (0-indexed). |
| `orig_x`, `orig_y` | **Ground Truth:** Exact pixel coordinates of the piece's top-left corner in the original image. |
| `sheet_x`, `sheet_y` | Storage location of the sprite in `jigsaw_spritesheet.png`. |
| `edge_top`, `edge_right`... | Morphology of edges (`TAB`, `SLOT`, `FLAT`). Useful for geometric matching. |
| `sprite_width`, `sprite_height` | Visual dimensions of the piece including tabs. |