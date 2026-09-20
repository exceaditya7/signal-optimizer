# Smart Traffic Optimizer

An intelligent computer vision system for real-time vehicle detection, tracking, and traffic flow density estimation using YOLOv11 and ByteTrack.

## 🚀 Features

- **Vehicle Detection**: Classifies Cars, Motorcycles, Buses, and Trucks using YOLO model inference.
- **Custom Region of Interest (ROI)**: Interactive GUI tool (`roi_test.py`) allowing users to draw and save custom polygon boundaries.
- **Multi-Object Tracking**: Uses ByteTrack tracking to monitor unique vehicle IDs crossing or staying within the designated ROI.
- **Dynamic Real-Time Stats**: Displays vehicle counts, density, and tracking markers overlay in real-time.

## 📁 Project Structure

```
smart-traffic-optimizer/
├── backend/
│   ├── detector.py          # Basic vehicle detection script
│   ├── roi_test.py          # Interactive ROI selector GUI tool
│   ├── traffic_analyzer.py  # Full ROI traffic analysis and vehicle tracking script
│   └── yolo11n.pt           # Pre-trained YOLO weights
├── data/
│   ├── roi_points.npy       # Saved ROI polygon coordinates
│   └── sample_videos/       # Input sample video files
├── requirements.txt         # Project Python dependencies
└── README.md                # Project documentation
```

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/exceaditya7/signal-optimizer.git
   cd smart-traffic-optimizer
   ```

2. **Set up Virtual Environment**:
   ```bash
   python3 -m venv backend/venv
   source backend/venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚥 Usage Instructions

### 1. Select Region of Interest (ROI)
Launch the interactive ROI selector tool to define your ROI polygon on a video frame:
```bash
python backend/roi_test.py
```
- **Controls**:
  - `LEFT CLICK`: Add polygon point (min 3 points).
  - `U`: Undo last point.
  - `R`: Reset points.
  - `S`: Save ROI points to `data/roi_points.npy`.
  - `Q`: Quit tool.

### 2. Run Smart Traffic Analyzer
Run real-time vehicle tracking and counting inside the defined ROI:
```bash
python backend/traffic_analyzer.py
```

### 3. Run Basic Vehicle Detector
Run full-frame vehicle detection without ROI filtering:
```bash
python backend/detector.py
```

## 📄 License
MIT License
