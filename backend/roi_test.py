import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Dynamic Path Resolution
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DEFAULT_VIDEO = PROJECT_ROOT / "data" / "sample_videos" / "vd.mp4"
FALLBACK_VIDEO = PROJECT_ROOT / "data" / "sample_videos" / "road_trafifc.mp4"
DEFAULT_ROI_FILE = PROJECT_ROOT / "data" / "roi_points.npy"

MAX_POINTS = 20

# Global points container for mouse callback
points = []


def mouse_callback(event, x, y, flags, param):
    """Mouse callback to record polygon points on left click."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < MAX_POINTS:
            points.append((x, y))
            print(f"Point {len(points):02d}: ({x}, {y})")
        else:
            print(f"Maximum limit of {MAX_POINTS} points reached.")


def select_roi(video_path: str = None, roi_output_path: str = None):
    """Interactive GUI tool for selecting a Region of Interest (ROI) polygon."""
    global points
    points.clear()

    if video_path is None:
        if DEFAULT_VIDEO.exists():
            video_path = str(DEFAULT_VIDEO)
        elif FALLBACK_VIDEO.exists():
            video_path = str(FALLBACK_VIDEO)
        else:
            print(f"ERROR: Sample video not found at {DEFAULT_VIDEO} or {FALLBACK_VIDEO}")
            return

    if roi_output_path is None:
        roi_output_path = str(DEFAULT_ROI_FILE)

    print(f"Opening video for ROI selection: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video file '{video_path}'.")
        return

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame from video.")
        cap.release()
        return

    window_name = "Smart Traffic Optimizer - ROI Selector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- ROI Selector Controls ---")
    print("  LEFT CLICK : Add point")
    print("  U          : Undo last point")
    print("  R          : Reset all points")
    print("  S          : Save ROI points")
    print("  Q          : Quit without saving\n")

    while True:
        display = frame.copy()

        # Draw selected points
        for i, (px, py) in enumerate(points):
            cv2.circle(display, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(
                display,
                str(i + 1),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

        # Draw lines between consecutive points
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(display, points[i], points[i + 1], (0, 255, 255), 3)

        # Close polygon and draw filled overlay when 3+ points exist
        if len(points) >= 3:
            cv2.line(display, points[-1], points[0], (0, 255, 255), 3)

            overlay = display.copy()
            polygon = np.array(points, dtype=np.int32)

            cv2.fillPoly(overlay, [polygon], (0, 255, 255))
            display = cv2.addWeighted(overlay, 0.15, display, 0.85, 0)
            cv2.polylines(display, [polygon], True, (0, 255, 255), 3)

        # Instruction HUD Panel
        cv2.rectangle(display, (0, 0), (430, 95), (0, 0, 0), -1)
        cv2.putText(
            display,
            f"Points: {len(points)}/{MAX_POINTS}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )
        cv2.putText(
            display,
            "CLICK = Add | U = Undo | R = Reset",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )
        cv2.putText(
            display,
            "S = Save | Q = Quit",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("r"):
            points.clear()
            print("ROI Reset.")

        elif key == ord("u"):
            if points:
                removed = points.pop()
                print(f"Removed point: {removed}")
            else:
                print("No points to remove.")

        elif key == ord("s"):
            if len(points) < 3:
                print("ERROR: Select at least 3 points before saving.")
                continue

            os.makedirs(os.path.dirname(roi_output_path), exist_ok=True)
            roi_array = np.array(points, dtype=np.int32)
            np.save(roi_output_path, roi_array)

            print("\n" + "=" * 40)
            print("ROI SAVED SUCCESSFULLY")
            print("=" * 40)
            print(f"Points saved: {len(points)}")
            print(f"File path: {roi_output_path}")
            print("\nCoordinates:")
            for idx, pt in enumerate(points, 1):
                print(f"  {idx:02d}: {pt}")
            print("=" * 40)
            break

        elif key == ord("q"):
            print("\nROI selection cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive ROI Selector Tool")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save output .npy ROI file")
    args = parser.parse_args()

    select_roi(args.video, args.output)