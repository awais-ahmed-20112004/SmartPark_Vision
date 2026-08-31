"""
SmartPark-Vision - YOLOv8 Vehicle Counter
=========================================
Pre-trained Ultralytics YOLOv8n detector, finalized version.

Improvement over naive counting: instead of only counting vehicles, we map each
detection's CENTER POINT into a parking-slot ROI with cv2.pointPolygonTest, so we
report per-slot occupancy (Green = Free, Red = Occupied) — directly comparable to
the classical method.

- 'yolov8n.pt' downloads automatically on first run.
- imgsz=1280 keeps full resolution so small aerial cars are not lost.
- CONF = 0.10 (tuned): the aerial PKLot feed shows cars as small/low-confidence
  objects, so 0.45 missed them; 0.25 recovers them at a small precision cost.

COCO class ids: 2 = car, 5 = bus, 7 = truck.
"""

import time

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "data/parking_lot.mp4"   # <-- change to your video path
MODEL = "yolov8n.pt"
CONF = 0.10                           # tuned (aerial cars are low-confidence)
IMGSZ = 1280
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

# ---- ROI grid parameters (same as pixel_counter.py) ----
X_START, X_END = 0, 1030
N_COLS = 22
Y_BANDS = [(0, 95), (100, 215), (215, 330), (330, 445), (445, 560)]


def make_row_rois(x_start, x_end, y_start, y_end, n_cols):
    xs = np.linspace(x_start, x_end, n_cols + 1)
    return [
        np.array([[int(xs[i]), y_start], [int(xs[i+1]), y_start],
                  [int(xs[i+1]), y_end], [int(xs[i]), y_end]], dtype=np.int32)
        for i in range(n_cols)
    ]


def point_in_any_roi(cx, cy, rois):
    for i, roi in enumerate(rois):
        if cv2.pointPolygonTest(roi, (float(cx), float(cy)), False) >= 0:
            return i
    return -1


def yolo_occupancy(model, frame, rois, conf=CONF, imgsz=IMGSZ, debug=False):
    results = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    occupied = [False] * len(rois)
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        idx = point_in_any_roi(cx, cy, rois)
        if idx >= 0:
            occupied[idx] = True
            detections.append((x1, y1, x2, y2, VEHICLE_CLASSES[cls], float(box.conf[0])))
    if debug:
        print(f"Raw boxes: {len(results.boxes)} | vehicle-class boxes: {len(detections)}")
    return occupied, detections


def main():
    model = YOLO(MODEL)
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
        occupied, detections = yolo_occupancy(model, frame, rois)
        fps = 1.0 / (time.time() - t0)

        free = 0
        for roi, occ in zip(rois, occupied):
            color = (0, 0, 255) if occ else (0, 255, 0)
            if not occ:
                free += 1
            cv2.polylines(frame, [roi], True, color, 2)

        for (x1, y1, x2, y2, label, dconf) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        cv2.rectangle(frame, (8, 8), (300, 78), (0, 0, 0), -1)
        cv2.putText(frame, f"Free slots: {free}/{len(rois)}", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"yolo | FPS: {fps:.1f}", (16, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YOLOv8 Vehicle Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
