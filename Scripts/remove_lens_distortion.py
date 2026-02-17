"""
Undistort Elios 3 grayscale images (equidistant fisheye model).
"""
import cv2
import yaml
import numpy as np
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

YAML_PATH = PROJECT_ROOT / "Data" / "YAML Files"
IMG_PATH = PROJECT_ROOT / "Data" / "ExtractedImages"
UNDISTORTED_IMG_PATH = PROJECT_ROOT / "Data" / "UndistortedImages"

# ---------- Calibration Files ----------
calib_files = {
    "camera_0": YAML_PATH / "camera_calibration_0.yaml",
    "camera_1": YAML_PATH / "camera_calibration_1.yaml",
    "camera_2": YAML_PATH / "camera_calibration_2.yaml",
}

# ---------- Load Calibration ----------
def load_calibration(yaml_path):
    with open(yaml_path, "r") as f:
        calib = yaml.safe_load(f)

    K = np.array(calib["camera_matrix"]["data"]).reshape(3, 3)
    D = np.array(calib["distortion_coefficients"]["data"])
    model = calib["distortion_model"]

    return K, D, model


# ---------- Undistortion ----------
def undistort_folder(camera_name, K, D, model):

    input_dir = IMG_PATH / f"{camera_name}_image_raw_compressed"
    output_dir = UNDISTORTED_IMG_PATH / camera_name
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(input_dir.glob("*.png"))
    if not images:
        print(f"No images found in {input_dir}")
        return

    sample = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    height, width = sample.shape[:2]

    print(f"{camera_name}")
    print("Image shape:", sample.shape)
    print("Distortion model:", model)
    print("D length:", len(D))
    print("-" * 40)

    K = K.astype(np.float64)
    D = D.astype(np.float64)

    if model == "equidistant":
        # Fisheye model (Elios 3)
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (width, height), np.eye(3), balance=1.0
        )

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), newK, (width, height), cv2.CV_16SC2
        )
    else:
        # Standard plumb_bob model
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 0)

        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, newK, (width, height), cv2.CV_16SC2
        )

    for im_path in images:
        im = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
        undistorted = cv2.remap(im, map1, map2, cv2.INTER_LINEAR)
        cv2.imwrite(str(output_dir / im_path.name), undistorted)

    print(f"{camera_name}: {len(images)} images undistorted.\n")


# ---------- Main ----------
def main():
    for camera_name, yaml_path in calib_files.items():
        print(f"Processing {camera_name}...")
        K, D, model = load_calibration(yaml_path)
        undistort_folder(camera_name, K, D, model)


if __name__ == "__main__":
    main()
