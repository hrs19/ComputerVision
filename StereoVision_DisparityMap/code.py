import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy
from scipy import signal as sig
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
    w_o_nms = []
    for y in range(size, h - size):
        for x in range(size, w - size):
            w_o_nms.append((x,y))
            if img[y, x] > threshold and img[y, x] == np.max(img[y - size:y + size + 1, x - size:x + size + 1]):
                maxima.append((x, y))

    return maxima, w_o_nms

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
        pt1 = (corners1[match[0]][0], corners1[match[0]][1])
        pt2 = (corners2[match[1]][0] + img1.shape[1], corners2[match[1]][1])
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(new_img, pt1, 3, color, -1)
        cv2.circle(new_img, pt2, 3, color, -1)
        cv2.line(new_img, pt1, pt2, color, 2)

    plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    cast_path = "Data/cast"
    cone_path = "Data/cone"
    
    path = 1
    if path == 1:
        path = cast_path
    else:
        path = cone_path
    
    file_li = []
    for file in os.listdir(path):
        file_li.append(file)

    img1 = cv2.imread(os.path.join(path, file_li[1]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(path, file_li[2]), cv2.IMREAD_GRAYSCALE)

    ksize = 3
    k = 0.04
    patch_size = 11
    ncc_threshold = 0.6
    harris_threshold = 1000

    corners1 = cv2.goodFeaturesToTrack(img1, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners1 = np.int0(corners1)
    corners1 = [(corner[0][0], corner[0][1]) for corner in corners1]

    corners2 = cv2.goodFeaturesToTrack(img2, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners2 = np.int0(corners2)
    corners2 = [(corner[0][0], corner[0][1]) for corner in corners2]

    r1 = compute_harris_r(img1, ksize, k)
    maxima1, _ = non_max_suppression(r1, 7, harris_threshold)

    r2 = compute_harris_r(img2, ksize, k)
    maxima2, _ = non_max_suppression(r2, 7, harris_threshold)

    matches = find_correspondences(img1, maxima1, img2, maxima2, patch_size, ncc_threshold)
    plot_correspondences(img1, maxima1, img2, maxima2, matches)

if __name__ == "__main__":
    main()
