import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

# Dynamic Path Resolution
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_PATH = BASE_DIR / "yolo11n.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = "yolo11n.pt"  # Fallback to YOLO auto-download/lookup

DEFAULT_VIDEO = PROJECT_ROOT / "data" / "sample_videos" / "vd.mp4"
FALLBACK_VIDEO = PROJECT_ROOT / "data" / "sample_videos" / "road_trafifc.mp4"

# COCO Vehicle Class IDs
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


def run_detector(video_path: str = None):
    """Run basic YOLO vehicle detection on a video stream."""
    if video_path is None:
        if DEFAULT_VIDEO.exists():
            video_path = str(DEFAULT_VIDEO)
        elif FALLBACK_VIDEO.exists():
            video_path = str(FALLBACK_VIDEO)
        else:
            print(f"ERROR: Video file not found at {DEFAULT_VIDEO} or {FALLBACK_VIDEO}")
            return

    print(f"Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source '{video_path}'.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Run YOLO inference
        results = model(frame, verbose=False)

        counts = {name: 0 for name in VEHICLE_CLASSES.values()}

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id in VEHICLE_CLASSES and confidence > 0.40:
                    vehicle_name = VEHICLE_CLASSES[class_id]
                    counts[vehicle_name] += 1

        annotated_frame = results[0].plot()
        total_vehicles = sum(counts.values())

        # Display counts on overlay
        y_offset = 40
        for name, count in counts.items():
            cv2.putText(
                annotated_frame,
                f"{name}s: {count}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            y_offset += 35

        cv2.putText(
            annotated_frame,
            f"TOTAL VEHICLES: {total_vehicles}",
            (20, y_offset + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

        cv2.imshow("Smart Traffic Optimizer - Detector", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else None
    run_detector(video)