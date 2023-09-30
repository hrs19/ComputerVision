import cv2
import numpy as np
import os
import random
from scipy.linalg import svd

def sobel_filter(img, axis):
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    if axis == 'x':
        kernel = np.transpose(kernel)
    filtered = np.zeros_like(img, dtype=np.float32)
    img = np.pad(img, 1, mode='edge')
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            filtered[i-1, j-1] = np.sum(kernel * img[i-1:i+2, j-1:j+2])
    return filtered

def filter_img(image, kernel):
    k_height, k_width = kernel.shape
    k_pad = k_height // 2
    img_height, img_width = image.shape
    output = np.zeros_like(image)
    for y in range(k_height // 2, img_height - k_height // 2):
        for x in range(k_width // 2, img_width - k_width // 2):
            roi = image[y-k_height//2:y+k_height//2+1, x-k_width//2:x+k_width//2+1]
            filtered_pixel = (roi * kernel).sum()
            output[y, x] = filtered_pixel
    return output

def compute_harris_r(img, ksize, k):
    dx = sobel_filter(img, axis='x')
    dy = sobel_filter(img, axis='y')
    dx2 = dx ** 2
    dy2 = dy ** 2
    dxy = dx * dy
    w = np.ones((ksize, ksize), np.float64)
    sdx2 = filter_img(dx2, w)
    sdy2 = filter_img(dy2, w)
    sdxy = filter_img(dxy, w)
    det = sdx2 * sdy2 - sdxy ** 2
    trace = sdx2 + sdy2
    r = det - k * trace ** 2
    return r

def non_max_suppression(img, size, threshold):
    h, w = img.shape
    maxima = []
    for y in range(size, h - size):
        for x in range(size, w - size):
            if img[y, x] > threshold and img[y, x] == np.max(img[y - size:y + size + 1, x - size:x + size + 1]):
                maxima.append((x, y))
    return maxima

def find_correspondences(img1, corners1, img2, corners2, patch_size, ncc_threshold):
    correspondences = []
    for i in range(len(corners1)):
        max_ncc = -1
        best_match = None
        for j in range(len(corners2)):
            patch1 = img1[corners1[i][1]-patch_size//2 : corners1[i][1]+patch_size//2,
                          corners1[i][0]-patch_size//2 : corners1[i][0]+patch_size//2]
            patch2 = img2[corners2[j][1]-patch_size//2 : corners2[j][1]+patch_size//2,
                          corners2[j][0]-patch_size//2 : corners2[j][0]+patch_size//2]
            ncc = np.sum((patch1 - np.mean(patch1)) * (patch2 - np.mean(patch2))) / \
                  (np.sqrt(np.sum((patch1 - np.mean(patch1)) ** 2)) * np.sqrt(np.sum((patch2 - np.mean(patch2)) ** 2)))
            if ncc > max_ncc and ncc > ncc_threshold:
                max_ncc = ncc
                best_match = (i, j)
        if best_match is not None:
            correspondences.append(best_match)
    return correspondences

def plot_correspondences(img1, corners1, img2, corners2, matches, title='Correspondences'):
    h, w = max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1]
    new_img = np.zeros((h, w, 3), dtype=np.uint8)
    new_img[:img1.shape[0], :img1.shape[1]] = img1
    new_img[:img2.shape[0], img1.shape[1]:] = img2
    for i, match in enumerate(matches):
        pt1 = tuple(corners1[match[0]])
        pt2 = (corners2[match[1]][0]+img1.shape[1],corners2[match[1]][1])
        color = tuple([random.randint(0, 255) for _ in range(3)])
        cv2.circle(new_img, pt1, 2, color, -1)
        cv2.circle(new_img, pt2, 2, color, -1)
        cv2.line(new_img, pt1, pt2, color, 1)
    cv2.imshow(title, new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    img1_path = 'image1.jpg'
    img2_path = 'image2.jpg'

    # Load the images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Define parameters
    ksize = 3
    k = 0.04
    corner_threshold = 10000
    patch_size = 21
    ncc_threshold = 0.9

    # Compute Harris corner response
    r1 = compute_harris_r(img1, ksize, k)
    r2 = compute_harris_r(img2, ksize, k)

    # Find corner points
    corners1 = non_max_suppression(r1, size=3, threshold=corner_threshold)
    corners2 = non_max_suppression(r2, size=3, threshold=corner_threshold)

    # Find correspondences
    correspondences = find_correspondences(img1, corners1, img2, corners2, patch_size, ncc_threshold)

    # Plot correspondences
    plot_correspondences(img1, corners1, img2, corners2, correspondences)

if __name__ == "__main__":
    main()
