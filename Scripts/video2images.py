import os
import cv2

from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = PROJECT_ROOT / "Data" / "Video"
RGB_IMG_PATH = PROJECT_ROOT / "Data" / "RGBImages"

video_path = VIDEO_PATH / '149_0002.MOV'

os.makedirs(RGB_IMG_PATH,  exist_ok=True)
video =cv2.VideoCapture(video_path)

if not video.isOpened():
    print('Failed to open video')
    exit()

fps = video.get(cv2.CAP_PROP_FPS)
print(f"Detected FPS: {fps}")

frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"Detected frame count: {frame_count}")      # 27203.0

duration = frame_count / fps
print(f"Detected Duration: {duration}")            # 906.6315000000001

frame_num = 0
while True:
    success, frame = video.read()
    if not success:
        break

    if fps and fps > 0:
        t_ms_idx = ((frame_num + 0.5)/fps) * 1000.0
    else:
        t_ms_idx = float('nan')

    filename = os.path.join(RGB_IMG_PATH, f"rgb_{frame_num:05d}.png")

    ok = cv2.imwrite(filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        print(f" failed to save {filename}")
    frame_num += 1
video.release()
print("Done")

