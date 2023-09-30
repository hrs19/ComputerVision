import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.linalg import svd

def sift_feat(images):
    sift = cv2.SIFT_create(nfeatures=90000)
    kps = []
    descs = []

    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        kps.append(kp)
        descs.append(des)

    return kps, descs

def KLT(images, kps):
    lk_params = dict(winSize=(21, 21),
                     maxLevel=100,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                     minEigThreshold=1e-5
                    )


    prev_img = images[0]
    prev_points = np.array([np.float32([kp.pt]) for kp in kps[0]])

    trackers = [prev_points]

    for next_img in images[1:]:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, next_img, prev_points, None, **lk_params)
        trackers.append(next_points)
        prev_img = next_img
        prev_points = next_points

    return trackers

def SFM(trackers):
    frm = len(trackers)
    pts = len(trackers[0])

    W = np.zeros((2 * frm, pts))

    for i, frame in enumerate(trackers):
        for j, point in enumerate(frame):
            W[2 * i, j] = point[0, 0]
            W[2 * i + 1, j] = point[0, 1]

    Wc = W - np.mean(W, axis=1, keepdims=True)

    U, S, Vt = svd(Wc)

    S = Vt[:3, :]

    return S

def gen_ply_file(points, filename):
    print(points.shape[1])
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(points.shape[1]))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')

        for pt in points.T:
            f.write(f'{pt[0]} {pt[1]} {pt[2]}\n')

if __name__=="__main__":

    images = [cv2.imread(f'hotel/{_}',0) for _ in os.listdir('hotel')]

    kps, descs = sift_feat(images)
    trackers = KLT(images, kps)
    S = SFM(trackers)

    gen_ply_file(S, 'output_3.ply')

