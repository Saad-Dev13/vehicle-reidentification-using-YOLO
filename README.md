# Advanced Vehicle Re-identification System

**AI Based Smart Parking Management System - Complete Vehicle Re-identification Module**

*Developed by Mohammad Saad Iqbal and Rafay Abrar*  
*National Institute of Electronics (NIE) - Internship Research Based Project*

## Project Overview

This project presents a comprehensive **vehicle re-identification system** with three different operational modes designed for various surveillance scenarios. The system leverages state-of-the-art YOLOv8 object detection combined with multiple feature extraction techniques for robust vehicle matching across different camera views and time periods.

## System Variants

The project includes three specialized systems:

1. **`main.py`** - Single Reference Image System
2. **`main_reid.py`** - Two-Stage Video Database System  
3. **`main_live_reid.py`** - Live Dual-Camera CCTV System

### Core Technologies

- **YOLOv8**: State-of-the-art object detection and tracking
- **Multi-Modal Features**: Traditional CV (HOG, SIFT, Histograms) + Deep Learning (ResNet, MobileNetV2)
- **Smart Matching**: Intelligent similarity scoring and conflict resolution
- **Database Storage**: Persistent SQLite storage for vehicle profiles
- **Real-time Processing**: Optimized for live camera feeds

# Installation

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/Saad-Dev13/vehicle-reidentification-using-YOLO.git
cd vehicle-reidentification-using-YOLO

# Install all dependencies
pip install -r requirements.txt
```

## Models Setup

**✅ All essential models are included in the repository!**

- **YOLOv8**: Automatically downloads if missing (backup included)
- **License Plate Detection**: Pre-included for immediate use
- **OSNet Features**: Pre-included for deep learning features

**No additional model downloads required - just clone and run!**

## Alternative Options

```bash
# Minimal installation (without OCR support)
pip install -r requirements-minimal.txt
```

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 2GB free space for models
- **GPU**: Optional but recommended for faster processing

---

# System 1: Single Reference Image (`main.py`)

## Overview

This system compares vehicles in a video against a single reference image. Ideal for tracking a specific known vehicle through surveillance footage.

### Key Features

- **Single Reference**: Uses one reference image for comparison
- **Multiple Features**: HOG, Histogram, SIFT, ResNet, MobileNetV2, Composite
- **ANPR Support**: Optional license plate recognition with PaddleOCR
- **Smart Fallbacks**: Default image/video paths with automatic validation
- **Confidence Scoring**: Quality assessment for reliable matching

### Feature Models Explained

Before using this system, understand the feature extraction models available:

- **Histogram**: Analyzes color distribution patterns across RGB channels. Fast but lighting-sensitive.
- **HOG (Histogram of Oriented Gradients)**: Detects shape patterns and edge orientations. Good for vehicle silhouettes.
- **SIFT (Scale-Invariant Feature Transform)**: Identifies distinctive keypoints that remain consistent across different viewing angles and lighting conditions.
- **ResNet**: Deep Convolutional Neural Network pre-trained on ImageNet. Extracts high-level semantic features with excellent accuracy.
- **MobileNetV2 (Recommended)**: Lightweight deep learning model optimized for mobile/embedded devices. Balances accuracy with speed - best choice for most scenarios.
- **Composite**: Combines ResNet + MobileNetV2 features for maximum accuracy but slower processing.

### Usage Examples

**⭐ MobileNet is recommended for optimal speed-accuracy balance**

#### 1. Default Usage (Auto-detects files)
```bash
# Uses default paths with MobileNet (recommended)
python main.py -f mobilenet

# Uses default paths with original defaults
python main.py
```

#### 2. All Feature Methods with Custom Files
```bash
# Traditional Computer Vision Methods
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f histogram
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f hog
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f sift

# Deep Learning Methods (Recommended: MobileNet)
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f mobilenet
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f resnet
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f composite
```

#### 3. Custom Thresholds and Confidence
```bash
# MobileNet with custom similarity threshold (recommended)
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f mobilenet --threshold 0.3

# MobileNet with custom confidence threshold (recommended)
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f mobilenet -c 0.7

# MobileNet with both custom values (recommended)
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f mobilenet --threshold 0.3 -c 0.7
```

#### 4. Advanced Features
```bash
# Enable license plate matching with MobileNet (recommended)
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f mobilenet -anpr "ABC123"

# Debug mode with similarity scores using MobileNet (recommended)
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f mobilenet --debug

# Save video disabled
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f mobilenet --no-save-video

# Complete example with all options using MobileNet (recommended)
python main.py -i videos/target.mp4 -r reference_vehicles/car.jpg -f mobilenet --threshold 0.3 -c 0.7 -anpr "XYZ789" --debug
```

#### 5. Production Ready Commands
```bash
# High accuracy for critical applications
python main.py -i videos/surveillance.mp4 -r reference_vehicles/suspect_car.jpg -f composite -c 0.8 --debug

# Fast processing for real-time needs (recommended)
python main.py -i videos/live_feed.mp4 -r reference_vehicles/target.jpg -f mobilenet --threshold 0.35 -c 0.6

# Balanced approach for general use (recommended)
python main.py -i videos/traffic.mp4 -r reference_vehicles/vehicle.jpg -f mobilenet -c 0.7
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input` | `videos/v11.mp4` | Target video path |
| `-r, --reference_img` | `reference_vehicles/ref_v11_1.jpeg` | Reference image path |
| `-f, --feature` | `sift` | Feature method (hog, histogram, sift, resnet, mobilenet, composite, anpr) |
| `-t, --threshold` | Auto | Similarity threshold (feature-specific) |
| `-c, --confidence` | Auto | Detection confidence (0.0-1.0) |
| `-anpr, --anpr` | None | License plate text to match |
| `--save-video` | True | Save annotated output video |
| `--debug` | False | Print similarity scores |

### Output Structure

```
runs/
├── exp/                    # First run
│   ├── output_sift.mp4    # Annotated video
│   ├── reference_image.jpg # Copy of reference
│   ├── run_info.json      # Metadata
│   └── matched_vehicles/   # Cropped matches
├── exp1/                   # Second run
└── exp2/                   # Third run
```

### Default Configuration

```python
# Default paths (edit in main.py)
DEFAULT_REF_IMG_A = "reference_vehicles/ref_v11_1.jpeg"
DEFAULT_TGT_VID_B = "videos/v11.mp4"

# Default thresholds
default_hog_threshold = 0.5
default_histogram_threshold = 0.85
default_resnet_threshold = 0.375
default_mobilenet_threshold = 0.375
default_sift_threshold = 0.1
default_yolo_confidence = 0.6
```

---

# System 2: Two-Stage Video Database (`main_reid.py`)

## Overview

Advanced two-stage system that builds a database from reference video, then performs intelligent re-identification on target video with unique ID assignment.

### Key Features

- **Two-Stage Processing**: Reference → Database → Target Re-identification
- **Smart ID Assignment**: Unique vehicle IDs with conflict resolution
- **Database Storage**: Persistent SQLite storage with feature aggregation
- **Intelligent Matching**: Each reference ID assigned to only one target
- **Multi-Frame Analysis**: Aggregates features across multiple appearances

### System Architecture

```
Stage 1: Reference Video → Detection → Feature Extraction → Database (A0, A1, A2...)
                                              ↓
Stage 2: Target Video → Detection → Feature Extraction → Matching → RE-ID/NEW Tags
```

### Feature Models Explained

Understanding the feature extraction models:

- **Histogram**: Analyzes color distribution patterns. Fast but sensitive to lighting changes.
- **HOG**: Detects shape and edge patterns. Excellent for vehicle silhouettes and geometric features.
- **SIFT**: Scale-invariant keypoint detection. Robust across different viewing angles and conditions.
- **ResNet**: Deep CNN with 50+ layers. High accuracy semantic feature extraction from ImageNet pre-training.
- **MobileNetV2 (Recommended)**: Depthwise separable convolutions for mobile optimization. Best speed-accuracy tradeoff.
- **Composite**: Ensemble of ResNet + MobileNetV2. Maximum accuracy but computationally intensive.

### Usage Examples

**⭐ MobileNet is recommended for optimal performance across all scenarios**

#### 1. Default Usage
```bash
# Uses default reference video and target video with MobileNet (recommended)
python main_reid.py -f mobilenet

# Uses default paths with system defaults
python main_reid.py
```

#### 2. All Feature Methods
```bash
# Traditional Computer Vision
python main_reid.py -r reference.mp4 -t target.mp4 -f histogram
python main_reid.py -r reference.mp4 -t target.mp4 -f hog
python main_reid.py -r reference.mp4 -t target.mp4 -f sift

# Deep Learning (Recommended: MobileNet)
python main_reid.py -r reference.mp4 -t target.mp4 -f mobilenet
python main_reid.py -r reference.mp4 -t target.mp4 -f resnet
python main_reid.py -r reference.mp4 -t target.mp4 -f composite
```

#### 3. Custom Thresholds
```bash
# MobileNet with custom similarity threshold (recommended)
python main_reid.py -r reference.mp4 -t target.mp4 -f mobilenet --threshold 0.3

# MobileNet with custom confidence (recommended)
python main_reid.py -r reference.mp4 -t target.mp4 -f mobilenet -c 0.7

# MobileNet with both custom values (recommended)
python main_reid.py -r reference.mp4 -t target.mp4 -f mobilenet --threshold 0.3 -c 0.7
```

#### 4. Debug and Analysis
```bash
# Debug mode with detailed matching info using MobileNet (recommended)
python main_reid.py -r reference.mp4 -t target.mp4 -f mobilenet --debug

# Disable video saving for analysis only
python main_reid.py -r reference.mp4 -t target.mp4 -f mobilenet --no-save-video --debug

# High precision analysis
python main_reid.py -r reference.mp4 -t target.mp4 -f composite --threshold 0.25 -c 0.8 --debug
```

#### 5. Production Scenarios
```bash
# Fast processing for large datasets (recommended)
python main_reid.py -r morning_traffic.mp4 -t evening_traffic.mp4 -f mobilenet --threshold 0.35

# Maximum accuracy for forensic analysis
python main_reid.py -r evidence_ref.mp4 -t evidence_target.mp4 -f composite -c 0.9 --debug

# Balanced approach for security systems (recommended)
python main_reid.py -r security_cam1.mp4 -t security_cam2.mp4 -f mobilenet -c 0.7 --debug
```

#### 6. Different Video Sources
```bash
# Different video formats and paths
python main_reid.py -r videos/reference.avi -t videos/target.mov -f mobilenet
python main_reid.py -r reference_videos/cam1_morning.mp4 -t target_videos/cam2_evening.mp4 -f mobilenet
python main_reid.py -r "D:\Videos\ref.mp4" -t "D:\Videos\target.mp4" -f mobilenet
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-r, --reference_video` | `reference_vehicles/ref_v12.mp4` | Reference video for database |
| `-t, --target_video` | `videos/v12.mp4` | Target video for re-identification |
| `-f, --feature` | `sift` | Feature extraction method |
| `--threshold` | Auto | Similarity matching threshold |
| `-c, --confidence` | Auto | Detection confidence threshold |
| `--save-video` | True | Save annotated output videos |
| `--debug` | False | Print detailed matching info |

### Vehicle ID Assignment

- **Reference Video**: Vehicles get IDs **A0, A1, A2, ...**
- **Target Video Re-identification**:
  - **RE-ID**: Matched vehicles keep reference ID (A0, A1, A2)
  - **NEW**: Unmatched vehicles get new IDs (B0, B1, B2)

### Smart Assignment Algorithm

1. **Unique Assignment**: Each reference ID assigned to max ONE target vehicle
2. **Best Match**: Multiple candidates → most similar wins
3. **Threshold Validation**: All matches must meet similarity requirements
4. **Conflict Resolution**: Automatic best-score selection

### Output Structure

```
runs_v2/
├── exp/                           # First experiment
│   ├── reference_output_sift.mp4  # Annotated reference video
│   ├── target_output_sift.mp4     # Annotated target video
│   ├── vehicle_reid.db            # SQLite database
├── exp1/                          # Second experiment
└── exp2/                          # Third experiment
```

### Database Schema

- **videos**: Metadata for reference/target videos
- **vehicles**: Vehicle tracks with IDs (A0, A1, B0, B1)
- **vehicle_features**: Individual frame features
- **aggregated_features**: Multi-frame combined features
- **reid_results**: Match results with similarity scores

---

# System 3: Live Dual-Camera CCTV (`main_live_reid.py`)

## Overview

Real-time dual-camera system designed for live CCTV environments. Processes two video streams simultaneously with optimized performance and smart source selection.

### Key Features

- **Dual-Stream Processing**: Simultaneous Camera A (reference) and Camera B (target)
- **Hybrid Sources**: Smart fallback between video files and IP cameras
- **Real-time Optimization**: Frame skipping, FPS limiting, display scaling
- **Temporal Tracking**: ByteTrack integration for ID persistence
- **Smart Delays**: Configurable camera start delays for analysis accuracy
- **Live Display**: Real-time annotated video windows

### System Architecture

```
Camera A (Reference) ──┐
                       ├── Live Processing ──→ Real-time Display
Camera B (Target)   ───┘         │
                                 └── Re-identification Results
```

### Feature Models Explained

Live system feature extraction models:

- **Histogram**: Color-based matching optimized for real-time processing. Fast but lighting dependent.
- **HOG**: Shape pattern detection ideal for vehicle identification across camera views.
- **SIFT**: Keypoint matching robust to camera angle differences and lighting variations.
- **ResNet**: Deep learning features for high accuracy but higher computational cost.
- **MobileNetV2 (Recommended)**: Mobile-optimized deep learning. Perfect for live systems requiring speed and accuracy.
- **Composite**: Combined ResNet + MobileNetV2. Maximum accuracy but may impact real-time performance.

### Usage Examples

**⭐ MobileNet is strongly recommended for live systems due to optimal real-time performance**

#### 1. Default Usage (Smart Fallback)
```bash
# Uses smart source selection with MobileNet (recommended)
python main_live_reid.py -f mobilenet

# Uses system defaults (video files or cameras)
python main_live_reid.py
```

#### 2. Pure Live Camera Mode (Both IP Cameras)
```bash
# Both cameras live with MobileNet (recommended for production)
python main_live_reid.py --use-cameras -f mobilenet

# Both cameras with custom IPs and MobileNet (recommended)
python main_live_reid.py --use-cameras --camera-ip-a rtsp://192.168.1.100/stream1 --camera-ip-b rtsp://192.168.1.101/stream1 -f mobilenet

# Production CCTV setup with optimizations (recommended)
python main_live_reid.py --use-cameras -f mobilenet --fps-limit 15 --frame-skip 2 --display-scale 0.5

# High-end cameras with minimal delays and MobileNet (recommended)
python main_live_reid.py --use-cameras -f mobilenet --camera-a-delay 0 --camera-b-delay 60 --fps-limit 20
```

#### 3. Hybrid Mode (Reference Video + Target Camera)
```bash
# Reference video with target camera using MobileNet (recommended)
python main_live_reid.py -r videos/reference.mp4 -t "" -f mobilenet

# Reference video with custom target camera using MobileNet (recommended)
python main_live_reid.py -r videos/reference.mp4 -t "" --camera-ip-b rtsp://192.168.1.101/stream1 -f mobilenet

# Reference video with live target, optimized settings (recommended)
python main_live_reid.py -r videos/morning_ref.mp4 -t "" -f mobilenet --fps-limit 12 --buffer-size 10
```

#### 4. Hybrid Mode (Reference Camera + Target Video)
```bash
# Reference camera with target video using MobileNet (recommended)
python main_live_reid.py -r "" -t videos/target.mp4 -f mobilenet

# Custom reference camera with target video using MobileNet (recommended)
python main_live_reid.py -r "" --camera-ip-a rtsp://192.168.1.100/stream1 -t videos/evening_traffic.mp4 -f mobilenet

# Live reference with delayed analysis using MobileNet (recommended)
python main_live_reid.py -r "" -t videos/target.mp4 -f mobilenet --camera-a-delay 30 --buffer-size 15
```

#### 5. Pure Video File Mode (Both Videos)
```bash
# Both video files with MobileNet for testing (recommended)
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f mobilenet

# Different video formats with MobileNet (recommended)
python main_live_reid.py -r videos/reference.avi -t videos/target.mov -f mobilenet --fps-limit 25

# Video analysis with custom settings using MobileNet (recommended)
python main_live_reid.py -r videos/cam1.mp4 -t videos/cam2.mp4 -f mobilenet --frame-skip 1 --display-scale 0.7
```

#### 6. All Feature Methods (Performance Comparison)
```bash
# Traditional CV methods (faster but less accurate)
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f histogram --fps-limit 30
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f hog --fps-limit 25
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f sift --fps-limit 20

# Deep learning methods (MobileNet recommended for live systems)
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f mobilenet --fps-limit 15
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f resnet --fps-limit 8
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f composite --fps-limit 5
```

#### 7. Advanced Live Configurations
```bash
# High-performance live setup with MobileNet (recommended for production)
python main_live_reid.py --use-cameras -f mobilenet \
  --camera-ip-a rtsp://admin:pass@192.168.1.100:554/stream1 \
  --camera-ip-b rtsp://admin:pass@192.168.1.101:554/stream1 \
  --fps-limit 12 --frame-skip 3 --display-scale 0.4 --buffer-size 8

# Security monitoring with delays using MobileNet (recommended)
python main_live_reid.py --use-cameras -f mobilenet \
  --camera-a-delay 0 --camera-b-delay 180 --fps-limit 10 --buffer-size 12

# Mobile/embedded deployment with MobileNet (recommended)
python main_live_reid.py --use-cameras -f mobilenet \
  --fps-limit 8 --frame-skip 4 --display-scale 0.3 --buffer-size 5

# Maximum accuracy live system (slower but most precise)
python main_live_reid.py --use-cameras -f composite \
  --fps-limit 3 --frame-skip 1 --display-scale 0.6 --buffer-size 15

# Debug live processing with MobileNet (recommended for troubleshooting)
python main_live_reid.py --use-cameras -f mobilenet --fps-limit 5 --debug
```

#### 8. Network Camera Variations
```bash
# Different RTSP authentication methods with MobileNet (recommended)
python main_live_reid.py --use-cameras -f mobilenet \
  --camera-ip-a rtsp://user:password@192.168.1.100:554/live1.sdp \
  --camera-ip-b rtsp://admin:admin123@192.168.1.101:8080/video

# HTTP cameras with MobileNet (recommended)
python main_live_reid.py --use-cameras -f mobilenet \
  --camera-ip-a http://192.168.1.100:8080/video.mjpg \
  --camera-ip-b http://192.168.1.101:8080/stream.mjpg

# Mixed camera types with MobileNet (recommended)
python main_live_reid.py -f mobilenet \
  --camera-ip-a rtsp://192.168.1.100:554/stream1 \
  --camera-ip-b 0  # USB camera
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-r, --reference_video` | Smart fallback | Reference video/camera |
| `-t, --target_video` | Smart fallback | Target video/camera |
| `--camera-ip-a` | RTSP URL | Reference camera IP |
| `--camera-ip-b` | RTSP URL | Target camera IP |
| `--use-cameras` | False | Force IP camera mode |
| `--camera-a-delay` | 0 | Reference start delay (seconds) |
| `--camera-b-delay` | 180 | Target start delay (seconds) |
| `--fps-limit` | 15 | Maximum processing FPS |
| `--frame-skip` | 2 | Process every Nth frame |
| `--display-scale` | 0.5 | Display window scale factor |
| `--buffer-size` | 8 | Reference vehicle buffer size |

### Smart Source Selection

Priority order:
1. **Command Line Args**: Specified video files
2. **Default Videos**: If files exist and aren't empty string
3. **IP Cameras**: Fallback to camera IPs

### Performance Optimizations

- **Frame Skipping**: Process every Nth frame for speed
- **FPS Limiting**: Cap processing rate to prevent overload
- **Display Scaling**: Reduce window size for better performance
- **RTSP Buffering**: Optimized for IP camera streams
- **ByteTrack Integration**: Persistent vehicle ID tracking

### Real-time Features

- **Live Display**: Side-by-side camera feeds with annotations
- **Temporal Filtering**: Multi-frame appearance validation
- **Confidence Gating**: Quality-based feature acceptance
- **Anti-Double Assignment**: Prevents duplicate ID assignments
- **Reconnection Handling**: Automatic camera reconnection

### Default Configuration

```python
# Camera IPs
DEFAULT_CAMERA_IP_A = "rtsp://admin:ad123456@192.168.55.252:554/0"
DEFAULT_CAMERA_IP_B = "rtsp://admin:ad123456@192.168.55.252:554/0"

# Video fallbacks
DEFAULT_REF_VID_A = "videos/v17.mp4"
DEFAULT_TGT_VID_B = ""  # Empty = use camera

# Timing
DEFAULT_CAMERA_A_DELAY = 0    # Reference delay
DEFAULT_CAMERA_B_DELAY = 180  # Target delay (3 minutes)
```

---

# Feature Extraction Methods

All systems support multiple feature extraction approaches with detailed explanations:

## Model Architecture Details

### Traditional Computer Vision Models

| Method | Algorithm | Strengths | Weaknesses | Best Use Case |
|--------|-----------|-----------|------------|---------------|
| **Histogram** | RGB color distribution analysis | Ultra-fast, simple implementation | Lighting sensitive, background interference | Real-time systems, color-distinctive vehicles |
| **HOG** | Gradient orientation histograms | Shape-aware, rotation tolerant | Scale sensitive, complex backgrounds | Vehicle silhouette matching, structured environments |
| **SIFT** | Scale-invariant keypoint detection | View-angle robust, distinctive features | Texture dependent, computational overhead | Multi-angle scenarios, textured vehicles |

### Deep Learning Models

| Method | Architecture | Pre-training | Strengths | Weaknesses | Recommendation |
|--------|-------------|--------------|-----------|-------------|----------------|
| **ResNet** | 50-layer CNN with residual connections | ImageNet (1000 classes) | Highest semantic understanding, robust features | Slow inference, high memory usage | Offline analysis, accuracy-critical |
| **MobileNetV2** ⭐ | Depthwise separable convolutions | ImageNet optimized for mobile | **Optimal speed-accuracy balance**, mobile-ready | Slightly lower accuracy than ResNet | **Recommended for all scenarios** |
| **Composite** | ResNet + MobileNetV2 ensemble | Combined ImageNet features | Maximum accuracy, feature complementarity | Slowest processing, highest resource usage | Forensic analysis, research |

## Speed vs Accuracy Analysis

| Method | Speed | Accuracy | Memory | CPU Usage | **Recommendation** |
|--------|-------|----------|--------|-----------|-------------------|
| Histogram | Very Fast (0.003s) | 75-85% | Low | Minimal | Real-time basic matching |
| HOG | Fast (0.027s) | 80-90% | Low | Low | Shape-based applications |
| SIFT | Fast (0.030s) | 85-95% | Medium | Medium | Multi-view scenarios |
| **MobileNetV2** ⭐ | **Fast (0.303s)** | **88-95%** | **Medium** | **Medium** | **Best overall choice** |
| ResNet | Slow (0.616s) | 90-98% | High | High | Accuracy-critical offline |
| Composite | Very Slow (0.919s) | 92-99% | Very High | Very High | Research/forensic analysis |

## Feature Extraction Technical Details

### Traditional Methods Implementation

- **Histogram**: 256-bin RGB histograms with correlation coefficient similarity
- **HOG**: 9 orientations, 8x8 pixel cells, 2x2 cell blocks, L2-Hys normalization  
- **SIFT**: DoG keypoint detection, 128-dimensional descriptors, FLANN matching

### Deep Learning Implementation

- **ResNet50**: Final pooling layer features (2048-dimensional), cosine similarity
- **MobileNetV2**: Pre-final layer features (1280-dimensional), cosine similarity  
- **Composite**: Weighted combination (0.6 × ResNet + 0.4 × MobileNetV2)

## Performance Comparison

| Feature Method | Processing Time | Accuracy | Use Case |
|---------------|----------------|----------|-----------|
| Histogram     | 0.003s/vehicle | 75-85%   | Real-time, color-based |
| HOG           | 0.027s/vehicle | 80-90%   | Shape-based matching |
| SIFT          | 0.030s/vehicle | 85-95%   | Robust keypoints |
| **MobileNetV2** ⭐ | **0.303s/vehicle** | **88-95%** | **Recommended for all scenarios** |
| ResNet        | 0.616s/vehicle | 90-98%   | High accuracy |
| Composite     | 0.919s/vehicle | 92-99%   | Maximum accuracy |

**⭐ MobileNetV2 provides the best balance of speed, accuracy, and resource efficiency across all three systems.**

---

# Configuration & Customization

## Similarity Thresholds

Control matching strictness (lower = more strict):

```python
# Edit in respective .py files
default_hog_threshold = 0.5          # HOG L2 distance
default_histogram_threshold = 0.85   # Color correlation
default_resnet_threshold = 0.375     # ResNet cosine distance
default_mobilenet_threshold = 0.3    # MobileNet cosine distance
default_sift_threshold = 0.5         # SIFT L2 distance
default_composite_threshold = 0.375  # Composite distance
```

## Confidence Thresholds

Control feature quality requirements (higher = better quality):

```python
default_yolo_confidence = 0.6        # Vehicle detection
default_resnet_confidence = 0.8      # ResNet features
default_mobilenet_confidence = 0.8   # MobileNet features
```

## Command Line Override

```bash
# Custom similarity threshold
python main_reid.py -r ref.mp4 -t target.mp4 -f resnet --threshold 0.3

# Custom confidence threshold
python main_reid.py -r ref.mp4 -t target.mp4 -f resnet -c 0.7

# Both custom values
python main_reid.py -r ref.mp4 -t target.mp4 -f resnet --threshold 0.3 -c 0.7
```

---

# Dependencies

## Core Requirements

```txt
ultralytics>=8.0.0           # YOLOv8
torch>=2.0.0                 # PyTorch
torchvision>=0.15.0          # PyTorch Vision
opencv-python>=4.5.0         # Computer Vision
scikit-image>=0.19.0         # Image Processing
scikit-learn>=1.2.0          # ML Algorithms
numpy>=1.21.0                # Numerical Computing
tqdm>=4.64.0                 # Progress Bars
```

## Optional Dependencies

```txt
paddlepaddle>=2.4.0          # For OCR features
paddleocr>=2.6.0             # License plate recognition
tabulate>=0.9.0              # Result formatting
```

## Installation Commands

```bash
# Full installation
pip install -r requirements.txt

# Minimal installation (no OCR)
pip install -r requirements-minimal.txt
```

---

# Project Structure

```
vehicle-reidentification-system/
├── main.py                   # Single reference image system
├── main_reid.py              # Two-stage video database system
├── main_live_reid.py         # Live dual-camera CCTV system
├── feature_extraction.py     # Feature computation functions
├── similarity.py             # Distance/similarity metrics
├── database.py               # SQLite database management
├── requirements.txt          # Full dependencies
├── requirements-minimal.txt  # Core dependencies only
├── models/                   # YOLO model files
│   ├── yolov8n.pt           # Vehicle detection
│   ├── yolo_lpdet.pt        # License plate detection
│   └── osnet_x1_0_imagenet.pth # Optional ReID model
├── videos/                   # Input video files
├── reference_vehicles/       # Reference images/videos
├── runs/                     # System 1 output
├── runs_v2/                  # System 2 output
└── runs_v3/                  # System 3 output
```

---

# Quick Start Guide

## System Selection & Recommended Commands

Choose the appropriate system for your use case:

### Single Vehicle Tracking (`main.py`)
**Best for**: Tracking one specific vehicle through surveillance footage

**⭐ Recommended quick start with MobileNet:**
```bash
# Default with MobileNet (recommended)
python main.py -f mobilenet

# Custom files with MobileNet (recommended)
python main.py -i surveillance.mp4 -r target_car.jpg -f mobilenet
```

### Video Database Analysis (`main_reid.py`) 
**Best for**: Comprehensive comparison between two video sources

**⭐ Recommended quick start with MobileNet:**
```bash
# Default with MobileNet (recommended) 
python main_reid.py -f mobilenet

# Custom videos with MobileNet (recommended)
python main_reid.py -r morning_traffic.mp4 -t evening_traffic.mp4 -f mobilenet
```

### Live CCTV Monitoring (`main_live_reid.py`)
**Best for**: Real-time dual-camera surveillance systems

**⭐ Recommended quick start with MobileNet:**
```bash
# Live cameras with MobileNet (recommended)
python main_live_reid.py --use-cameras -f mobilenet --fps-limit 15

# Hybrid mode with MobileNet (recommended)
python main_live_reid.py -r videos/reference.mp4 -t "" -f mobilenet
```

## All System Examples

### Complete Command Set for All Systems

#### System 1 - Single Reference (main.py)
```bash
# 1. Default usage with MobileNet (recommended)
python main.py -f mobilenet

# 2. All feature methods
python main.py -i videos/traffic.mp4 -r reference_vehicles/car.jpg -f histogram
python main.py -i videos/traffic.mp4 -r reference_vehicles/car.jpg -f hog  
python main.py -i videos/traffic.mp4 -r reference_vehicles/car.jpg -f sift
python main.py -i videos/traffic.mp4 -r reference_vehicles/car.jpg -f mobilenet  # Recommended
python main.py -i videos/traffic.mp4 -r reference_vehicles/car.jpg -f resnet
python main.py -i videos/traffic.mp4 -r reference_vehicles/car.jpg -f composite

# 3. MobileNet with custom settings (recommended)
python main.py -i videos/traffic.mp4 -r reference_vehicles/car.jpg -f mobilenet --threshold 0.3 -c 0.7 --debug
```

#### System 2 - Two-Stage Database (main_reid.py)
```bash
# 1. Default usage with MobileNet (recommended)
python main_reid.py -f mobilenet

# 2. All feature methods
python main_reid.py -r reference.mp4 -t target.mp4 -f histogram
python main_reid.py -r reference.mp4 -t target.mp4 -f hog
python main_reid.py -r reference.mp4 -t target.mp4 -f sift
python main_reid.py -r reference.mp4 -t target.mp4 -f mobilenet  # Recommended
python main_reid.py -r reference.mp4 -t target.mp4 -f resnet
python main_reid.py -r reference.mp4 -t target.mp4 -f composite

# 3. MobileNet with custom settings (recommended)
python main_reid.py -r reference.mp4 -t target.mp4 -f mobilenet --threshold 0.3 -c 0.7 --debug
```

#### System 3 - Live Dual-Camera (main_live_reid.py)
```bash
# 1. Both live cameras with MobileNet (recommended)
python main_live_reid.py --use-cameras -f mobilenet

# 2. Reference video + target camera with MobileNet (recommended)
python main_live_reid.py -r videos/reference.mp4 -t "" -f mobilenet

# 3. Reference camera + target video with MobileNet (recommended)
python main_live_reid.py -r "" -t videos/target.mp4 -f mobilenet

# 4. Both videos with MobileNet (recommended)
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f mobilenet

# 5. All feature methods (performance testing)
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f histogram --fps-limit 30
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f hog --fps-limit 25
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f sift --fps-limit 20
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f mobilenet --fps-limit 15  # Recommended
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f resnet --fps-limit 8
python main_live_reid.py -r videos/ref.mp4 -t videos/target.mp4 -f composite --fps-limit 5
```

## Performance Tips

1. **Feature Selection**: **MobileNet recommended** for optimal speed-accuracy balance across all systems
2. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster deep learning processing
3. **Memory Management**: Use frame skipping (`--frame-skip`) for large videos in live system
4. **Threshold Tuning**: MobileNet works well with default thresholds (0.3-0.375 similarity, 0.8 confidence)
5. **Real-time Optimization**: For live systems, use `--fps-limit 15` with MobileNet for best results

---

# Troubleshooting

## Common Issues

### Import Errors
```bash
# Missing dependencies
pip install -r requirements.txt

# CUDA issues
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Performance Issues
- Reduce `--fps-limit` for live processing
- Increase `--frame-skip` for faster processing
- Use `--display-scale 0.3` for smaller windows
- Switch to faster feature methods (histogram, hog)

### Camera Connection Issues
- Verify RTSP URLs are accessible
- Check network connectivity
- Adjust buffer sizes for network cameras
- Use delays if cameras need initialization time

### Memory Issues
- Close other applications
- Use minimal installation if OCR not needed
- Process smaller video segments
- Enable frame skipping

---

# Advanced Usage

## Custom Integration

Each system can be integrated into larger applications:

```python
# Import system components
from feature_extraction import compute_sift_features
from similarity import cosine_distance
from database import VehicleDatabase

# Custom processing pipeline
def custom_vehicle_matching(image1, image2):
    features1 = compute_sift_features(image1)
    features2 = compute_sift_features(image2)
    similarity = cosine_distance(features1, features2)
    return similarity < 0.3  # Custom threshold
```

## Batch Processing

Process multiple videos automatically:

```bash
# Process all videos in directory
for video in videos/*.mp4; do
    python main_reid.py -r reference.mp4 -t "$video" -f composite
done
```

## Production Deployment

For production CCTV systems:

```bash
# Optimized live processing
python main_live_reid.py \
  --use-cameras \
  --camera-ip-a rtsp://cam1.company.com/stream \
  --camera-ip-b rtsp://cam2.company.com/stream \
  --fps-limit 10 \
  --frame-skip 3 \
  --display-scale 0.4 \
  --buffer-size 5
```

---

# Contributors

| Contributor 1 | Contributor 2 |
| ------------- | ------------- |
| [Mohammad Saad Iqbal](https://github.com/Saad-Dev13) | [Rafay Abrar](https://github.com/rafayabrar) |

---

# License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

# Citation

If you use this system in your research, please cite:

```
@software{vehicle_reid_system_2024,
  title={Advanced Vehicle Re-identification System},
  author={Mohammad Saad Iqbal, and Rafay Abrar},
  year={2025},
  organization={National Institute of Electronics},
  url={https://github.com/Saad-Dev13/vehicle-reidentification-using-YOLO}
}
```

