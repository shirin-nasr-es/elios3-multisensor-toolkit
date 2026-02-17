"""
Undistort Elios RGB images (equidistant fisheye model).
"""
import os
import cv2
import yaml
import numpy as np
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMG_PATH = PROJECT_ROOT / "Data" / "RGBImages"
UNDISTORTED_IMG_PATH = PROJECT_ROOT / "Data" / "Undistorted_RGBImages"

if not UNDISTORTED_IMG_PATH.exists():
    UNDISTORTED_IMG_PATH.mkdir()

# ------ Calibration parameters ------
# Camera intrinsic matrix
K = np.array([[385.66647728473237, 0.0, 326.6261845137072],
              [0.0, 385.3608783941113, 197.44977467934152],
              [0.0, 0.0, 1.0]])

# Distortion Coefficients
D = np.array([-0.07467741495847284, -0.012604664419241502,  0.006387703817457847, -0.002776265668555331])

image_files = sorted([f for f in os.listdir(IMG_PATH) if f.endswith('.png')])

for fname in image_files:
    print('Processing', fname)
    im_path = os.path.join(IMG_PATH, fname)
    im = cv2.imread(im_path)

    h, w = im.shape[:2]
    scale_x = w / 640
    scale_y = h / 400

    K_new = K.copy()
    K_new[0, 0] *= scale_x
    K_new[0, 2] *= scale_x
    K_new[1, 1] *= scale_y
    K_new[1, 2] *= scale_y

    # Undistorting Procedure
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_new, D, np.eye(3), K_new, (w, h), cv2.CV_32FC1, cv2.CV_16SC2)
    undistorted_im = cv2.remap(im, map1, map2, cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(UNDISTORTED_IMG_PATH, fname), undistorted_im)