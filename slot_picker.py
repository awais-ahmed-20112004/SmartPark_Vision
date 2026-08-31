"""
SmartPark-Vision - Slot Picker
==============================
A beginner-friendly helper that lets you draw a polygon (ROI) around each
parking slot on a single video frame, then saves the coordinates to JSON.

Why we need this:
    A "Region of Interest" (ROI) is just the corners of the parking slot.
    Instead of hard-coding pixel values, we click them on screen once and
    reuse them forever.

How to use:
    1. Run:  python classical_detector/slot_picker.py data/parking_lot.mp4
    2. A window opens showing the first frame of the video.
    3. LEFT-CLICK around ONE parking slot to place its corners.
    4. Press 'n'  -> the current polygon is saved and you start a new slot.
    5. Repeat for every slot. Press 'q' (or Esc) when done.
    6. The coordinates are written to data/rois.json.

Controls:
    LEFT-CLICK  -> add a corner to the current polygon
    'n'         -> finish current polygon, start the next slot
    'z'         -> undo the last corner
    'q' / Esc   -> save everything and quit
"""

import argparse
import json
import os

import cv2
import numpy as np

# The corners of the polygon currently being drawn.
current_points = []


def click(event, x, y, flags, param):
    """Mouse callback: record a corner on left-click."""
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))


def draw_overlay(frame, slots):
    """Draw finished slots (blue) and the in-progress polygon (green)."""
    img = frame.copy()
    for slot in slots:
        cv2.polylines(img, [np.array(slot)], isClosed=True,
                      color=(255, 0, 0), thickness=2)
    if len(current_points) >= 2:
        cv2.polylines(img, [np.array(current_points)], isClosed=False,
                      color=(0, 255, 0), thickness=2)
    for pt in current_points:
        cv2.circle(img, pt, 4, (0, 0, 255), -1)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Select parking-slot ROIs from a video or image.")
    parser.add_argument("input", help="Path to the video file or a single image.")
    parser.add_argument("--output", default="data/rois.json",
                        help="Where to save the ROI coordinates.")
    args = parser.parse_args()

    # Load the first frame as a background image.
    cap = cv2.VideoCapture(args.input)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = cv2.imread(args.input)  # in case it's an image, not a video
    if frame is None:
        raise SystemExit(f"Could not read '{args.input}'. Check the path.")

    # Shrink very wide frames so the window fits on screen.
    h, w = frame.shape[:2]
    scale = 1.0
    if w > 1200:
        scale = 1200.0 / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    slots = []  # finished slots, each a list of [x, y] points

    cv2.namedWindow("Slot Picker")
    cv2.setMouseCallback("Slot Picker", click)
    print("LEFT-CLICK=add corner | n=next slot | z=undo | q=save & quit")

    while True:
        img = draw_overlay(frame, slots)
        cv2.imshow("Slot Picker", img)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('n'):  # finish this slot
            if len(current_points) >= 3:
                slots.append([list(p) for p in current_points])
                print(f"Slot {len(slots)} saved with {len(current_points)} corners.")
            else:
                print("A slot needs at least 3 corners - keep clicking.")
            current_points.clear()
        elif key == ord('z'):  # undo last corner
            if current_points:
                current_points.pop()
        elif key == ord('q') or key == 27:  # quit
            break

    cv2.destroyAllWindows()

    # Keep the last polygon if the user forgot to press 'n'.
    if len(current_points) >= 3:
        slots.append([list(p) for p in current_points])

    if not slots:
        raise SystemExit("No slots were selected - nothing to save.")

    # Convert back from the shrunk image to full-resolution coordinates.
    final_slots = []
    for slot in slots:
        pts = [[int(x / scale), int(y / scale)] for x, y in slot]
        final_slots.append(pts)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(final_slots, f, indent=2)

    print(f"Saved {len(final_slots)} slot ROIs to {args.output}")


if __name__ == "__main__":
    main()
