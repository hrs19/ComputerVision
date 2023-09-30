# Computer Vision Stereo Vision Project

This Python script is part of the Computer Vision I course project 3, which focuses on stereo vision. The project explores various aspects of stereo vision, including feature extraction, fundamental matrix estimation, disparity map computation, and visualizing the results.

## Dependencies

This script relies on several Python libraries, including:

- `cv2`: OpenCV for image processing.
- `numpy`: NumPy for numerical operations.
- `matplotlib`: Matplotlib for data visualization.
- `scipy`: SciPy for scientific computing.
- `random`: Python's built-in random library.

Make sure you have these libraries installed before running the script.

## Functions

Here are some of the key functions implemented in the script:

- `sobel_filter`: Applies a Sobel filter to an image along a specified axis.
- `filter_img`: Applies a custom filter to an image.
- `compute_harris_r`: Computes the Harris corner response for an image.
- `non_max_suppression`: Performs non-maximum suppression on an image to find corner keypoints.
- `find_correspondences`: Finds correspondences between two images using feature matching.
- `plot_correspondences`: Visualizes correspondences between two images.

## Usage

1. Set the appropriate file paths in the `main` function for your dataset.
2. Customize the parameters such as `ksize`, `k`, `patch_size`, and `ncc_threshold` based on your needs.
3. Run the script to perform stereo vision tasks, including feature extraction and correspondence matching.

## Data

The script assumes you have a dataset of stereo image pairs located in the specified file paths.