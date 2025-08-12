# Model Files

This directory contains the pre-trained models required for the vehicle re-identification system.

## Included Models

### 1. `yolov8n.pt` (6.23 MB)
- **Purpose**: Vehicle detection and tracking
- **Model**: YOLOv8 Nano - optimized for speed
- **Auto-download**: Yes, automatically downloaded by Ultralytics if missing
- **Usage**: Primary object detection for identifying vehicles in video streams

### 2. `yolo_lpdet.pt` (5.95 MB)
- **Purpose**: License plate detection (ANPR)
- **Model**: Custom YOLO model trained for license plates
- **Auto-download**: No, manually provided
- **Usage**: Optional license plate recognition feature

### 3. `osnet_x1_0_imagenet.pth` (10.41 MB)
- **Purpose**: Deep feature extraction for vehicle re-identification
- **Model**: OSNet (Omni-Scale Network) pre-trained on ImageNet
- **Auto-download**: No, manually provided
- **Usage**: Used by deep learning feature extraction methods

## Auto-Download Capability

The system is designed with automatic model downloading:

- **YOLOv8 models**: Automatically downloaded by Ultralytics library if missing
- **Custom models**: Included in repository for immediate functionality

## Model Loading Fallbacks

The code includes robust model loading with fallbacks:

```python
# YOLOv8 with automatic download fallback
try:
    if os.path.exists("models/yolov8n.pt"):
        self.yolo_model = YOLO("models/yolov8n.pt")
    else:
        self.yolo_model = YOLO("yolov8n.pt")  # Auto-download
except Exception:
    # Alternative loading methods
```

## Usage

Models are automatically loaded by the system scripts:
- `main.py` - Single reference system
- `main_reid.py` - Two-stage database system  
- `main_live_reid.py` - Live dual-camera system

No manual model setup required - just run the scripts!

## Model Sources

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **OSNet**: [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- **License Plate YOLO**: Custom trained model
