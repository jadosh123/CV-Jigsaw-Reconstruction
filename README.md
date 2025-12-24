# Computer Vision Jigsaw Reconstruction
A Python-based framework for generating, playing, and conceptually solving jigsaw puzzles using Computer Vision. This project bridges the gap between synthetic data generation and interactive gameplay, serving as a robust testbed for future AI solver implementations (SIFT/RANSAC, Deep Learning).

## Features
1. Procedural Puzzle Generation (generate_jigsaw.py)

    - Intelligent Cutting: Automatically segments any image into a jigsaw grid with varying edge types (Tab, Slot, Flat).

    - Bezier Curve Geometry: Uses mathematical Bezier curves to create  realistic, smooth puzzle piece contours.

    - Spritesheet Packing: Generates an optimized, transparent PNG spritesheet containing all pieces.

    - Metadata Manifest: Exports a detailed CSV (puzzle_data.csv) containing ground truth data for every piece (coordinates, edge types, dimensions), essential for AI training.

## Project Structure
```
CV-Jigsaw-Reconstruction/
├── img/
│   └── Grumpy_Kitty.png       # Source image for the puzzle
├── jigsaw/
│   ├── jigsaw_spritesheet.png # Generated transparent spritesheet
│   └── puzzle_data.csv        # Generated ground truth manifest
├── generate_jigsaw.py         # Script to cut image & create dataset
├── requirements.txt           # Dependencies
└── README.md
```

## Getting Started
**Prerequisites**

- Python3.7+
- OpenCV (opencv-python)
- Numpy

**run in the terminal**:
```
pip install opencv-python numpy
```

## 1. Generating a Puzzle
Run the generator to create a new puzzle from your source image.

```
python generate_jigsaw.py
```

- **Input:** Reads img/Grumpy_Kitty.png (or modify the script to point to your image).

- **Output:** Creates jigsaw/jigsaw_spritesheet.png and jigsaw/puzzle_data.csv.

- **Configuration:** You can adjust the grid size (e.g., 10x10) inside the __main__ block of generate_jigsaw.py.

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