"""
SmartPark-Vision - Classical Detector (Pixel Counter)
=====================================================
Occupancy detection using ONLY classical image processing (no deep learning).

Finalized version — matches the tested Colab pipeline.

Two key upgrades over the naive approach:
  1. PROGRAMMATIC ROI GRID: for a wide aerial parking view we generate a grid of
     slot rectangles (make_row_rois) instead of clicking each slot. The grid is
     still "custom slot ROI coordinates" defined by X_START/X_END, N_COLS and
     Y_BANDS.
  2. AREA-NORMALIZED THRESHOLD: because grid slots differ in size with aerial
     perspective, we compare pixel DENSITY (white_pixels / area) against the
     calibrated density THRESHOLD / 3000, rather than a raw pixel count.
     THRESHOLD still anchors the calibration (Roll #65369 -> 869).

Pipeline per frame:
    grayscale -> Gaussian blur -> adaptive threshold -> dilation ->
    per-ROI white-pixel count -> density vs threshold -> Free/Occupied -> HUD.
"""

import time

import cv2
import numpy as np

VIDEO_PATH = "data/parking_lot.mp4"   # <-- change to your video path

THRESHOLD = 869                       # calibrated cutoff (Roll #65369)
GAUSSIAN_KERNEL = (5, 5)
BLOCK_SIZE = 25
C = 16

# ---- ROI grid parameters (tune to your video) ----
X_START, X_END = 0, 1030              # exclude trees/road on the right
N_COLS = 22                           # spots per row band
Y_BANDS = [(0, 95), (100, 215), (215, 330), (330, 445), (445, 560)]


def make_row_rois(x_start, x_end, y_start, y_end, n_cols):
    xs = np.linspace(x_start, x_end, n_cols + 1)
    return [
        np.array([[int(xs[i]), y_start], [int(xs[i+1]), y_start],
                  [int(xs[i+1]), y_end], [int(xs[i]), y_end]], dtype=np.int32)
        for i in range(n_cols)
    ]


def preprocess(frame, blur_kernel=GAUSSIAN_KERNEL, block=BLOCK_SIZE, c=C):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, c)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)
    return gray, blurred, binary


def count_white(binary, roi):
    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [roi], 255)
    return cv2.countNonZero(cv2.bitwise_and(binary, mask))


def classical_occupancy(binary, rois, threshold):
    """Area-normalized occupancy: density > threshold/3000 => occupied."""
    occupied = []
    for roi in rois:
        area = cv2.contourArea(roi)
        px = count_white(binary, roi)
        occupied.append((px / max(area, 1)) > (threshold / 3000.0))
    return occupied


def main():
    rois = []
    for y0, y1 in Y_BANDS:
        rois += make_row_rois(X_START, X_END, y0, y1, N_COLS)
    print(f"{len(rois)} slot ROIs generated.")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {VIDEO_PATH}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        _, _, binary = preprocess(frame)
        occupied = classical_occupancy(binary, rois, THRESHOLD)
        fps = 1.0 / (time.time() - t0)

        free = 0
        for roi, occ in zip(rois, occupied):
            color = (0, 0, 255) if occ else (0, 255, 0)
            if not occ:
                free += 1
            cv2.polylines(frame, [roi], True, color, 2)

        cv2.rectangle(frame, (8, 8), (300, 78), (0, 0, 0), -1)
        cv2.putText(frame, f"Free slots: {free}/{len(rois)}", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"classical | FPS: {fps:.1f}", (16, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Classical Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
