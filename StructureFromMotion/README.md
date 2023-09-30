# StructureFromMotion Project

## Introduction

This Python script implements the Structure from Motion (SfM) technique using OpenCV. It processes a sequence of images from the CMU Hotel Sequence dataset to recover 3D structure from 2D features.

## Prerequisites

Before running the code, make sure you have the following software and packages installed on your system:

- Python (>=3.6)
- OpenCV (cv2)
- NumPy
- Matplotlib
- SciPy

You can install the required Python packages using pip:

```bash
pip install opencv-python numpy matplotlib scipy
```

## How to Use

1. Clone or download this repository to your local machine.

2. Place your input images in a directory named 'hotel' within the project directory.

3. Run the Python script 'code.py' using the following command:

```bash
python code.py
```

## Code Explanation

This Python script performs Structure from Motion (SfM) on a sequence of images from the CMU Hotel Sequence dataset. Below are the key functions and their explanations:

- **sift_feat(images):** This function detects SIFT features in a list of images and returns keypoints and descriptors. SIFT (Scale-Invariant Feature Transform) features are distinctive points in images.

- **KLT(images, kps):** The KLT function performs feature tracking using the Kanade-Lucas-Tomasi (KLT) tracker on a list of images with SIFT keypoints. It tracks these keypoints across the image sequence.

- **SFM(trackers):** The SFM function applies the factorization method to recover 3D structure from the tracked features. It returns 3D coordinates.

- **gen_ply_file(points, filename):** The gen_ply_file function generates a PLY file from the 3D structure points. PLY files are commonly used to represent 3D models.

## Output

The script processes the images, performs Structure from Motion, and generates a PLY file named 'result.ply' containing 3D coordinates. You can visualize the 3D structure results using software like MeshLab.
