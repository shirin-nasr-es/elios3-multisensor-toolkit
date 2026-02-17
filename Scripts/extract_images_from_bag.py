"""
Extract grayscale images from an Elios 3 ROS bag file.

The Elios 3 includes three grayscale tracking cameras, used for velocity estimation, in addition to the RGB camera.
The RGB stream does not contain reliable timestamps in the bag file, whereas the grayscale camera topics provide
accurate header timestamps.

These grayscale images are extracted to supply precise timing information for synchronization.
"""

import os
import cv2
import struct
import numpy as np

from pathlib import Path
from rosbags.rosbag1 import Reader

# ---------- Paths ----------
# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Build paths relative to the root
BAG_PATH = PROJECT_ROOT / "Data" / "BAG Files" / 'E3RDAA23250004_00227_149_1flight_obcbag.bag'
OUT_ROOT = PROJECT_ROOT / "Data" / "ExtractedImages"

camera_topics =[
    '/camera_0/image_raw/compressed',
    '/camera_1/image_raw/compressed',
    '/camera_2/image_raw/compressed',
]

MAX_FRAMES_PER_CAMERA = None

# Convert a ROS topic into a filesystem-safe folder name
def sanitize_topic(topic: str) -> str:
    return topic.strip("/").replace("/", "_")

def read_u32(buf: bytes, pos: int) -> tuple[int, int]:
    (val,) = struct.unpack_from("<I", buf, pos)  # little-endian uint32
    return val, pos + 4

def read_string(buf: bytes, pos: int) -> tuple[str, int]:
    length, pos = read_u32(buf, pos)
    s = buf[pos : pos + length].decode("utf-8", errors="replace")
    return s, pos + length


def parse_ros1_compressed_image(raw: bytes) -> tuple[int, int, str, str, bytes]:
    """
    Parse ROS1 sensor_msgs/CompressedImage from raw serialized bytes.

    ROS1 layout:
      uint32 seq
      time stamp: uint32 secs, uint32 nsecs
      string frame_id
      string format
      uint32 data_len + data bytes
    """
    pos = 0
    _seq, pos = read_u32(raw, pos)
    sec, pos = read_u32(raw, pos)
    nsec, pos = read_u32(raw, pos)
    frame_id, pos = read_string(raw, pos)
    fmt, pos = read_string(raw, pos)
    data_len, pos = read_u32(raw, pos)
    data = raw[pos : pos + data_len]
    return sec, nsec, frame_id, fmt, data


def main() -> None:
    if not BAG_PATH.exists():
        raise FileNotFoundError(f"Bag file not found:\n{BAG_PATH}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Output folders per topic
    topic_out = {}
    for t in camera_topics:
        folder = OUT_ROOT / sanitize_topic(t)
        folder.mkdir(parents=True, exist_ok=True)
        topic_out[t] = folder

    saved = {t: 0 for t in camera_topics}
    skipped_decode = {t: 0 for t in camera_topics}

    with Reader(str(BAG_PATH)) as reader:
        conns = [c for c in reader.connections if c.topic in camera_topics]
        if not conns:
            raise RuntimeError("Camera topics not found in bag. Check CAMERA_TOPICS list.")

        for conn, t_ns, rawdata in reader.messages(connections=conns):
            topic = conn.topic

            if MAX_FRAMES_PER_CAMERA is not None and saved[topic] >= MAX_FRAMES_PER_CAMERA:
                continue

            # Parse ROS1 CompressedImage manually (robust on Windows)
            try:
                sec, nsec, frame_id, fmt, img_bytes = parse_ros1_compressed_image(rawdata)
            except Exception:
                skipped_decode[topic] += 1
                continue

            # Decode JPEG/PNG bytes to grayscale
            buf = np.frombuffer(img_bytes, dtype=np.uint8)
            gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                skipped_decode[topic] += 1
                continue

            num = saved[topic]
            filename = f"frame_{num:06d}_{sec}_{nsec:09d}.png"
            out_path = topic_out[topic] / filename

            ok = cv2.imwrite(str(out_path), gray)
            if ok:
                saved[topic] += 1
            else:
                skipped_decode[topic] += 1

    print("\nDone.")
    for t in camera_topics:
        print(f"{t}: saved={saved[t]}  skipped={skipped_decode[t]}  ->  {topic_out[t]}")


if __name__ == "__main__":
    main()
