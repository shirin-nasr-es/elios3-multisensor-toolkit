"""
Synchronize RGB frames with camera_2 by assigning timestamps based on cam2 timing.
This is necessary because the RGB stream does not contain reliable timestamps.
"""

import os
import cv2
import math
import yaml
import numpy as np
import pandas as pd

from pathlib import Path

def rgb_to_gray(rgb_id):
    fps_rgb = float(29.97002997002997)
    fps_gray = float(30.016041)

    # Frame mapping slope
    slope = fps_gray / fps_rgb

    if rgb_id <= 55:
        gray_id = rgb_id + 105
    else:
        gray_id = 160 + (rgb_id - 55) * slope

    return math.floor(gray_id)


# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RGB_PATH = PROJECT_ROOT / "Data" / "Undistorted_RGBImages"
GRAY_PATH = PROJECT_ROOT / "Data" / "UndistortedImages"/"camera_2"

OUTPUT_RGB_PATH = PROJECT_ROOT / "Data" / "Timestamped_RGBImages"
CSV_PATH = PROJECT_ROOT / "Data" / "CSV Files"

if not OUTPUT_RGB_PATH.exists():
    OUTPUT_RGB_PATH.mkdir()

# Load image files
rgb_files = sorted([f for f in os.listdir(RGB_PATH) if f.endswith('.png')])
gray_files = sorted([i for i in os.listdir(GRAY_PATH) if i.endswith('.png')])

rgb_IDs = []
gray_IDs = []
timestamps_sec = []
timestamps_nsec = []
print(type(gray_files))
for  fname in rgb_files:
    im_path = os.path.join(RGB_PATH, fname)
    im = cv2.imread(im_path)
    frame_n = fname.split('_')[1]
    frame_n = frame_n.split('.')[0]

    frame_id = int(frame_n)
    gray_id = rgb_to_gray(frame_id)
    gray_name = gray_files[gray_id]
    print(frame_id, gray_id)
    #print(gray_name)

    timestamp_sec = gray_name.split('_')[2]
    timestamp_nsec = gray_name.split('_')[3].split('.')[0]

    rgb_new_name = f"rgb_{frame_id:05d}_"+ timestamp_sec+ "_"+ timestamp_nsec + ".png"
    #print(rgb_new_name)

    filename = os.path.join(OUTPUT_RGB_PATH, rgb_new_name)
    cv2.imwrite(filename, im)

    rgb_IDs.append(frame_id)
    gray_IDs.append(gray_id)
    timestamps_sec.append(timestamp_sec)
    timestamps_nsec.append(timestamp_nsec)


# Create Pandas Dataframe
time_df = pd.DataFrame()
time_df['rgb_ID'] = rgb_IDs
time_df ['Gray_id'] = gray_IDs
time_df['timestamp_sec'] = timestamps_sec
time_df['timestamp_nsec'] = timestamps_nsec

# Write into csv File
time_df.to_csv(CSV_PATH, index=False)











