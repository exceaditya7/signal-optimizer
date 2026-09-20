import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Dynamic Path Resolution
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DEFAULT_MODEL = BASE_DIR / "yolo11n.pt"
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "sample_videos" / "vd.mp4"
FALLBACK_VIDEO = PROJECT_ROOT / "data" / "sample_videos" / "road_trafifc.mp4"
DEFAULT_ROI = PROJECT_ROOT / "data" / "roi_points.npy"

# COCO Vehicle Classes
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


def load_roi(roi_path: Path) -> np.ndarray:
    """Load Region of Interest points array from .npy file."""
    if not roi_path.exists():
        print(f"\nERROR: ROI file not found at '{roi_path}'.")
        print("Please run ROI selection first using:")
        print("  python backend/roi_test.py\n")
        return None

    try:
        roi_points = np.load(roi_path)
        print(f"Loaded ROI points from: {roi_path}")
        return roi_points
    except Exception as e:
        print(f"ERROR: Failed to load ROI points from '{roi_path}': {e}")
        return None


def analyze_traffic(video_path: str = None, roi_path: str = None, model_path: str = None):
    """Run real-time vehicle detection, tracking, and ROI traffic analysis."""
    # Resolve video path
    if video_path is None:
        if DEFAULT_VIDEO.exists():
            video_path = str(DEFAULT_VIDEO)
        elif FALLBACK_VIDEO.exists():
            video_path = str(FALLBACK_VIDEO)
        else:
            print(f"ERROR: Sample video not found at {DEFAULT_VIDEO} or {FALLBACK_VIDEO}")
            return

    # Resolve ROI path
    if roi_path is None:
        roi_points = load_roi(DEFAULT_ROI)
    else:
        roi_points = load_roi(Path(roi_path))

    if roi_points is None:
        return

    # Resolve Model path
    if model_path is None:
        model_path = str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else "yolo11n.pt"

    print("--------------------------------")
    print("SMART TRAFFIC ANALYZER")
    print("--------------------------------")
    print(f"Model Path : {model_path}")
    print(f"Video Path : {video_path}")
    print(f"ROI File   : {roi_path or DEFAULT_ROI}")
    print("--------------------------------")

    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file '{video_path}'.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Resolution: {width} x {height} @ {fps:.1f} FPS")

    window_name = "Smart Traffic Analyzer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    previous_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video analysis completed.")
            break

        # Run ByteTrack multi-object tracking
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )
        result = results[0]

        # Draw ROI polygon outline and translucent fill
        cv2.polylines(frame, [roi_points], True, (0, 255, 255), 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_points], (0, 255, 255))
        frame = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)

        counts = {name: 0 for name in VEHICLE_CLASSES.values()}

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id not in VEHICLE_CLASSES or confidence < 0.40:
                    continue

                if box.id is None:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Check if vehicle center point is inside ROI
                inside_roi = cv2.pointPolygonTest(roi_points, (center_x, center_y), False)
                if inside_roi < 0:
                    continue

                vehicle_name = VEHICLE_CLASSES[class_id]
                counts[vehicle_name] += 1

                # Draw vehicle bounding box and track ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{vehicle_name} ID:{track_id} {confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                previous_positions[track_id] = (center_x, center_y)

        total_vehicles = sum(counts.values())

        # Display HUD vehicle statistics
        cv2.rectangle(frame, (5, 5), (280, 180), (0, 0, 0), -1)
        y_offset = 30
        for name, count in counts.items():
            cv2.putText(
                frame,
                f"{name}s: {count}",
                (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )
            y_offset += 30

        cv2.putText(
            frame,
            f"Vehicles in ROI: {total_vehicles}",
            (15, y_offset + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Analysis finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Traffic Analyzer with ROI Tracking")
    parser.add_argument("--video", type=str, default=None, help="Path to input video")
    parser.add_argument("--roi", type=str, default=None, help="Path to ROI .npy file")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO model (.pt)")
    args = parser.parse_args()

    analyze_traffic(args.video, args.roi, args.model)