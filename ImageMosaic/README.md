# Image Feature Matching using Harris Corner Detection and NCC

### Need to fix this code, it was written adhoc for submission
This Python script demonstrates feature matching between two images using Harris Corner Detection and Normalized Cross-Correlation (NCC). It identifies corner points in both images and finds correspondences between them.

## Requirements

- OpenCV (`cv2`)
- NumPy (`numpy`)
- SciPy (`scipy.linalg.svd`)
- Random (`random`)

## Description

This script performs the following tasks:

1. Computes Harris Corner Response: It calculates the Harris corner response for both input images to detect corner points.

2. Finds Corner Points: It identifies corner points in the images using non-maximum suppression.

3. Matches Features: It finds correspondences between corner points in the two images based on NCC.

4. Visualizes Correspondences: It visualizes the correspondences by drawing lines between matching corner points.

## Usage

To use the script, follow these steps:

1. Prepare your two images (`img1` and `img2`) and specify their paths in the `img1_path` and `img2_path` variables.

2. Customize the parameters as needed:
   - `ksize`: The size of the Sobel filter kernel.
   - `k`: The Harris Corner constant.
   - `corner_threshold`: Threshold for corner point detection.
   - `patch_size`: Size of the patch for NCC computation.
   - `ncc_threshold`: Threshold for NCC matching.

3. Run the script by executing the `main()` function.

4. The script will display the correspondences between the two images, allowing you to visually inspect the feature matching.
