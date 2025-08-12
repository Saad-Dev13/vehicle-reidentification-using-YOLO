"""
@brief: Vehicle Re-identification System using YOLOv8 and Multiple Feature Extractors
@project: AI Based Smart Parking Management System - Vehicle Re-identification Module
@authors: Mohammad Saad Iqbal, Rafay Abrar
@organization: National Institute of Electronics (NIE)
@description: Advanced vehicle re-identification system supporting multiple feature extraction methods
             including ResNet, MobileNetV2, SIFT, HOG, color histograms, and composite approaches
             with confidence-based validation and intelligent fallback mechanisms.
"""

# Fix SSL certificate issues for downloading models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import os
import argparse
import glob
import json
from datetime import datetime
from timeit import default_timer
from ultralytics import YOLO
from torchvision import models
# from torchreid.utils import FeatureExtractor  # Temporarily disabled due to compatibility issues
from feature_extraction import *
from similarity import *
from paddleocr import PaddleOCR
from tqdm import tqdm
import sys

# Default sources (fallbacks when CLI is empty)
DEFAULT_REF_IMG_A = "reference_vehicles/ref_v11_1.jpeg"  # Reference image default
DEFAULT_TGT_VID_B = "videos/v11.mp4"                               # Target/input video default

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default=None, help="path to input/target video (falls back to DEFAULT_TGT_VID_B)")
parser.add_argument("-r", "--reference_img", default=None, help="path to reference image (falls back to DEFAULT_REF_IMG_A)")
parser.add_argument("-f", "--feature", default="sift", choices=["hog", "histogram", "sift", "resnet", "mobilenet", "composite", "anpr"],
                    help="Feature type to use for comparison")
parser.add_argument("-t", "--threshold", type=float, default=None, help="similarity threshold (feature-specific)")
parser.add_argument("-c", "--confidence", type=float, default=None, help="detection confidence threshold (0.0-1.0)")
parser.add_argument("-anpr", "--anpr", default=None, type=str, help="number plate to match")
parser.add_argument("--save-video", action="store_true", help="save output video (default: True)")
parser.add_argument("--no-save-video", dest="save_video", action="store_false", help="don't save output video")
parser.add_argument("--debug", action="store_true", help="print similarity scores for debugging")
parser.set_defaults(save_video=True)
args = parser.parse_args()

# Resolve sources with smart fallback
def resolve_sources():
    inp = args.input or DEFAULT_TGT_VID_B
    ref = args.reference_img or DEFAULT_REF_IMG_A

    inp = os.path.normpath(inp) if inp else inp
    ref = os.path.normpath(ref) if ref else ref

    missing = []
    if not inp or not os.path.exists(inp):
        missing.append(f"Input video not found: {inp}")
    if not ref or not os.path.exists(ref):
        missing.append(f"Reference image not found: {ref}")

    if missing:
        print("❌ Input error:")
        for m in missing:
            print(f"   • {m}")
        print("💡 Set defaults in code (DEFAULT_REF_IMG_A / DEFAULT_TGT_VID_B) or pass --input / --reference_img")
        sys.exit(1)

    args.input = inp
    args.reference_img = ref

    print("✅ SMART SOURCE CONFIGURATION:")
    print(f"   • Reference Image: {args.reference_img}")
    print(f"   • Target Video:   {args.input}")

resolve_sources()

def create_runs_directory():
    """Create runs directory with incremental experiment naming"""
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)
    
    # Find existing experiments
    existing_exps = glob.glob(os.path.join(runs_dir, "exp*"))
    if not existing_exps:
        exp_name = "exp"
    else:
        # Extract numbers from existing experiments
        exp_nums = []
        for exp in existing_exps:
            exp_base = os.path.basename(exp)
            if exp_base == "exp":
                exp_nums.append(0)
            elif exp_base.startswith("exp") and exp_base[3:].isdigit():
                exp_nums.append(int(exp_base[3:]))
        
        next_num = max(exp_nums) + 1 if exp_nums else 1
        exp_name = f"exp{next_num}"
    
    exp_dir = os.path.join(runs_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir, exp_name

def print_header():
    """Print a clean header for the application"""
    print("=" * 80)
    print("🚗 ADVANCED VEHICLE RE-IDENTIFICATION SYSTEM")
    print("    AI Based Smart Parking Management System")
    print("    National Institute of Electronics (NIE)")
    print("    Developed by: Mohammad Saad Iqbal & Rafay Abrar")
    print("=" * 80)
    print(f"📅 Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_config(args, exp_name, exp_dir):
    """Print configuration details"""
    print("📋 CONFIGURATION")
    print("-" * 40)
    print(f"🎯 Experiment: {exp_name}")
    print(f"📁 Output Directory: {exp_dir}")
    print(f"🎬 Input Video: {args.input}")
    print(f"🖼️  Reference Image: {args.reference_img}")
    print(f"🔧 Feature Method: {args.feature.upper()}")
    
    # Show actual threshold being used
    if args.feature == "hog":
        threshold = args.threshold if args.threshold is not None else default_hog_threshold
        print(f"🎚️  Similarity Threshold: {threshold}")
    elif args.feature == "histogram":
        threshold = args.threshold if args.threshold is not None else default_histogram_threshold
        print(f"🎚️  Similarity Threshold: {threshold}")
    elif args.feature == "resnet":
        threshold = args.threshold if args.threshold is not None else default_resnet_threshold
        print(f"🎚️  Cosine Distance Threshold: {threshold}")
    elif args.feature == "mobilenet":
        threshold = args.threshold if args.threshold is not None else default_mobilenet_threshold
        print(f"🎚️  MobileNet Cosine Distance Threshold: {threshold}")
    elif args.feature == "composite":
        threshold = args.threshold if args.threshold is not None else default_composite_threshold
        print(f"🎚️  Composite Distance Threshold: {threshold}")
    elif args.feature == "sift":
        threshold = args.threshold if args.threshold is not None else default_sift_threshold
        print(f"🎚️  SIFT Match Ratio Threshold: {threshold}")
    
    # Show confidence thresholds
    confidence = args.confidence if args.confidence is not None else default_yolo_confidence
    print(f"🎯 YOLO Detection Confidence: {confidence}")
    
    # Show feature extraction confidence thresholds when applicable
    if args.feature == "resnet":
        resnet_conf = args.confidence if args.confidence is not None else default_resnet_confidence
        print(f"🎯 ResNet Feature Confidence: {resnet_conf}")
    elif args.feature == "mobilenet":
        mobilenet_conf = args.confidence if args.confidence is not None else default_mobilenet_confidence
        print(f"🎯 MobileNet Feature Confidence: {mobilenet_conf}")
    elif args.feature == "composite":
        resnet_conf = args.confidence if args.confidence is not None else default_resnet_confidence
        mobilenet_conf = args.confidence if args.confidence is not None else default_mobilenet_confidence
        print(f"🎯 ResNet Feature Confidence: {resnet_conf}")
        print(f"🎯 MobileNet Feature Confidence: {mobilenet_conf}")
    
    if args.anpr:
        lp_conf = args.confidence if args.confidence is not None else default_lp_confidence
        print(f"🎯 License Plate Detection Confidence: {lp_conf}")
    
    if args.anpr:
        print(f"🔍 License Plate: {args.anpr}")
    print(f"💾 Save Video: {'Yes' if args.save_video else 'No'}")
    if args.debug:
        print(f"🐛 Debug Mode: Enabled")
    print()

def save_run_info(exp_dir, args, results_summary):
    """Save run information to JSON file"""
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "input_video": args.input,
        "reference_image": args.reference_img,
        "feature_method": args.feature,
        "anpr_plate": args.anpr,
        "results": results_summary
    }
    
    info_path = os.path.join(exp_dir, "run_info.json")
    with open(info_path, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    return info_path

# Create experiment directory
exp_dir, exp_name = create_runs_directory()

# Similarity thresholds - more reasonable defaults
default_hog_threshold = 0.5
default_histogram_threshold = 0.85
default_resnet_threshold = 0.375  # Much more reasonable for ResNet features
default_mobilenet_threshold = 0.375  # MobileNetV2 cosine distance threshold
default_sift_threshold = 0.1    # More lenient for SIFT matching
default_composite_threshold = ((default_resnet_threshold + default_mobilenet_threshold)/2)

# Confidence thresholds for different models
default_yolo_confidence = 0.6  # YOLO detection confidence
default_lp_confidence = 0.5    # License plate detection confidence
default_resnet_confidence = 0.8    # ResNet feature quality threshold (optional)
default_mobilenet_confidence = 0.8  # MobileNet feature quality threshold (optional)

# Print application header and configuration
print_header()
print_config(args, exp_name, exp_dir)

# Use command-line threshold if provided, otherwise use defaults
hog_threshold = args.threshold if args.threshold is not None else default_hog_threshold
histogram_threshold = args.threshold if args.threshold is not None else default_histogram_threshold
cosine_threshold = args.threshold if args.threshold is not None else default_resnet_threshold
mobilenet_threshold = args.threshold if args.threshold is not None else default_mobilenet_threshold
sift_threshold = args.threshold if args.threshold is not None else default_sift_threshold
composite_threshold = args.threshold if args.threshold is not None else default_composite_threshold

# Use command-line confidence if provided, otherwise use defaults
yolo_confidence = args.confidence if args.confidence is not None else default_yolo_confidence
lp_confidence = args.confidence if args.confidence is not None else default_lp_confidence
resnet_confidence = args.confidence if args.confidence is not None else default_resnet_confidence
mobilenet_confidence = args.confidence if args.confidence is not None else default_mobilenet_confidence

print("🔧 INITIALIZING MODELS")
print("-" * 40)

# Initialize YOLO model
print("📦 Loading YOLOv8 model...")
model = YOLO("models/yolov8n.pt")

# Initialize video capture
print("🎬 Loading input video...")
cap = cv2.VideoCapture(args.input)

if not cap.isOpened():
    print("❌ Error: Could not open video file!")
    sys.exit(1)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"📊 Video Properties: {width}x{height} @ {fps}fps, {total_frames} frames")

# Setup video writer if saving video
video_writer = None
if args.save_video:
    output_video_path = os.path.join(exp_dir, f"output_{args.feature}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print(f"💾 Output video will be saved to: {output_video_path}")

# Directory for saving images of matched vehicles
save_dir = os.path.join(exp_dir, "matched_vehicles")
os.makedirs(save_dir, exist_ok=True)

print(f"🖼️  Matched vehicle images will be saved to: {save_dir}")
print()

# Load reference vehicle image and resize
print("🖼️  LOADING REFERENCE IMAGE")
print("-" * 40)
reference_image_path = args.reference_img
reference_image = cv2.imread(reference_image_path)
if reference_image is None:
    print("❌ Error: Reference image not found!")
    sys.exit(1)

print(f"✅ Reference image loaded successfully")

# Save reference image to experiment directory
ref_copy_path = os.path.join(exp_dir, "reference_image.jpg")
cv2.imwrite(ref_copy_path, reference_image)

if args.feature == "hog" or args.feature == "histogram":
    # Standardize image size for feature extraction
    standard_size = (128, 64)  # Typical size used for HOG feature extraction
    # Resize reference image
    reference_image = cv2.resize(reference_image, standard_size)

print()
print("🧠 EXTRACTING REFERENCE FEATURES")
print("-" * 40)

start_time = default_timer()
# Compute features for the reference image
if args.feature == "hog":
    print("🔍 Computing HOG features...")
    reference_hog_features = compute_hog_features(reference_image)
elif args.feature == "histogram":
    print("🎨 Computing color histogram...")
    reference_color_histogram = compute_color_histogram(reference_image)
elif args.feature =="sift":
    print("🔬 Computing SIFT features...")
    sift = cv2.SIFT_create()
    reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
elif args.feature == "resnet":
    print("🧠 Loading ResNet model and extracting features...")
    # Load a pre-trained ResNet model
    model_resnet = models.resnet50(pretrained=True)
    model_resnet.eval()  # Set model to evaluation mode
    reference_dnn_features, reference_confidence = extract_dnn_features(model_resnet, reference_image, resnet_confidence)
    print(f"🎯 Reference ResNet feature confidence: {reference_confidence:.3f}")
    if reference_confidence < resnet_confidence:
        print(f"⚠️  Warning: Reference image confidence ({reference_confidence:.3f}) below threshold ({resnet_confidence})")
elif args.feature == "mobilenet":
    print("📱 Loading MobileNetV2 model and extracting features...")
    # Load a pre-trained MobileNetV2 model
    model_mobilenet = models.mobilenet_v2(pretrained=True)
    model_mobilenet.eval()  # Set model to evaluation mode
    reference_mobilenet_features, reference_mobilenet_conf = extract_dnn_features(model_mobilenet, reference_image, mobilenet_confidence)
    print(f"🎯 Reference MobileNet feature confidence: {reference_mobilenet_conf:.3f}")
    if reference_mobilenet_conf < mobilenet_confidence:
        print(f"⚠️  Warning: Reference image confidence ({reference_mobilenet_conf:.3f}) below threshold ({mobilenet_confidence})")
elif args.feature == "composite":
    print("🔧 Loading Composite features (ResNet + MobileNetV2)...")
    # Load both ResNet and MobileNetV2 models
    model_resnet = models.resnet50(pretrained=True)
    model_resnet.eval()
    model_mobilenet = models.mobilenet_v2(pretrained=True)
    model_mobilenet.eval()
    
    # Extract features from both models with confidence
    reference_resnet_features, reference_resnet_conf = extract_dnn_features(model_resnet, reference_image, resnet_confidence)
    reference_mobilenet_features, reference_mobilenet_conf = extract_dnn_features(model_mobilenet, reference_image, mobilenet_confidence)
    
    print(f"🎯 Reference ResNet confidence: {reference_resnet_conf:.3f}")
    print(f"🎯 Reference MobileNet confidence: {reference_mobilenet_conf:.3f}")
    
    if reference_resnet_conf < resnet_confidence:
        print(f"⚠️  Warning: ResNet confidence ({reference_resnet_conf:.3f}) below threshold ({resnet_confidence})")
    if reference_mobilenet_conf < mobilenet_confidence:
        print(f"⚠️  Warning: MobileNet confidence ({reference_mobilenet_conf:.3f}) below threshold ({mobilenet_confidence})")
# elif args.feature == "osnet":
#     print("🎯 Loading OSNet model and extracting features...")
#     # OSNet temporarily disabled due to torchreid compatibility issues
#     pass
# elif args.feature == "composite":
#     print("🔧 Loading Composite features (SIFT + OSNet)...")
#     # Composite temporarily disabled due to torchreid compatibility issues
#     pass

# ANPR OCR
if args.anpr:
    print("🔍 Initializing ANPR system...")
    ocr = PaddleOCR(lang='en',rec_algorithm='CRNN')
    lp_detector = YOLO("models/yolo_lpdet.pt")

end_time = default_timer()
reference_feature_time = end_time - start_time
print(f"✅ Reference features extracted in {reference_feature_time:.3f}s")
print()

# Initialize tracking variables
frame_count = 0
matches_found = 0
total_detections = 0
processing_times = []

print("🎬 PROCESSING VIDEO")
print("-" * 40)

# get vehicle classes: bicycle, car, motorcycle, airplane, bus, train, truck, boat
# class_list = [1, 2, 3, 4, 5, 6, 7, 8]
class_list = [2]

# Progress bar for video processing
pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame", 
           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

# Tracking and matching
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    pbar.update(1)
    
    frame_start_time = default_timer()
    results = model.track(frame, show=False, verbose=False, conf=yolo_confidence, classes=class_list, persist=True)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()  # Use xyxy format
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        annotated_frame = results[0].plot()
        total_detections += len(boxes)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Use xyxy format
            crop_img = frame[y1:y2, x1:x2]  # Use xyxy format

            # resize image for non-DNN features
            if args.feature == "hog" or args.feature == "histogram":
                crop_img = cv2.resize(crop_img, standard_size)  # Resize to standard size

            match = False
            similarity_score = None
            # time for feature extraction
            start_time = default_timer()
            if args.feature == "hog":
                # Compare HOG features
                hog_features = compute_hog_features(crop_img)
                l2_dist = l2_distance(hog_features, reference_hog_features)
                similarity_score = l2_dist
                match = l2_dist < hog_threshold
                if args.debug and frame_count % 30 == 0:  # Print every 30th frame to avoid spam
                    print(f"🐛 HOG L2 Distance: {l2_dist:.4f} (threshold: {hog_threshold})")
            elif args.feature == "histogram":
                # Compare color histograms
                color_histogram = compute_color_histogram(crop_img)
                hist_dist = histogram_distance(color_histogram, reference_color_histogram)
                similarity_score = hist_dist
                match = hist_dist > histogram_threshold
                if args.debug and frame_count % 30 == 0:
                    print(f"🐛 Histogram Distance: {hist_dist:.4f} (threshold: {histogram_threshold})")
            elif args.feature=="sift":
                #Compare SIFT features
                keypoints, descriptors = sift.detectAndCompute(crop_img, None)
                if descriptors is not None and reference_descriptors is not None:
                    matches = matcher.knnMatch(reference_descriptors, descriptors, k=2)
                    # Apply ratio test
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                    # Calculate match ratio
                    match_ratio = len(good_matches) / len(reference_descriptors) if len(reference_descriptors) > 0 else 0
                    similarity_score = match_ratio
                    match = match_ratio > sift_threshold
                    if args.debug and frame_count % 30 == 0:
                        print(f"🐛 SIFT Match Ratio: {match_ratio:.4f} (threshold: {sift_threshold})")
            elif args.feature == "resnet":
                # Compare DNN features with confidence validation
                dnn_features, feature_confidence = extract_dnn_features(model_resnet, crop_img, resnet_confidence)
                
                # Only proceed if feature confidence is acceptable
                if feature_confidence >= resnet_confidence:
                    resnet_cosine_dist = cosine_distance(dnn_features, reference_dnn_features)
                    similarity_score = resnet_cosine_dist
                    match = resnet_cosine_dist < cosine_threshold
                    if args.debug and frame_count % 30 == 0:
                        print(f"🐛 ResNet: Confidence={feature_confidence:.3f}, Distance={resnet_cosine_dist:.4f} (threshold: {cosine_threshold})")
                else:
                    # Low confidence features - skip matching
                    similarity_score = 1.0  # Max distance (no match)
                    match = False
                    if args.debug and frame_count % 30 == 0:
                        print(f"🐛 ResNet: Low confidence ({feature_confidence:.3f} < {resnet_confidence}) - skipping match")
            elif args.feature == "mobilenet":
                # Compare MobileNetV2 features with confidence validation
                mobilenet_features, feature_confidence = extract_dnn_features(model_mobilenet, crop_img, mobilenet_confidence)
                
                # Only proceed if feature confidence is acceptable
                if feature_confidence >= mobilenet_confidence:
                    mobilenet_cosine_dist = cosine_distance(mobilenet_features, reference_mobilenet_features)
                    similarity_score = mobilenet_cosine_dist
                    match = mobilenet_cosine_dist < mobilenet_threshold
                    if args.debug and frame_count % 30 == 0:
                        print(f"🐛 MobileNet: Confidence={feature_confidence:.3f}, Distance={mobilenet_cosine_dist:.4f} (threshold: {mobilenet_threshold})")
                else:
                    # Low confidence features - skip matching
                    similarity_score = 1.0  # Max distance (no match)
                    match = False
                    if args.debug and frame_count % 30 == 0:
                        print(f"🐛 MobileNet: Low confidence ({feature_confidence:.3f} < {mobilenet_confidence}) - skipping match")
            elif args.feature == "composite":
                # Compare composite features (ResNet + MobileNetV2) with confidence validation
                resnet_features, resnet_feat_conf = extract_dnn_features(model_resnet, crop_img, resnet_confidence)
                mobilenet_features, mobilenet_feat_conf = extract_dnn_features(model_mobilenet, crop_img, mobilenet_confidence)
                
                # Check if both features meet confidence requirements
                resnet_valid = resnet_feat_conf >= resnet_confidence
                mobilenet_valid = mobilenet_feat_conf >= mobilenet_confidence
                
                if resnet_valid and mobilenet_valid:
                    # Both features are confident - proceed with composite matching
                    resnet_cosine_dist = cosine_distance(resnet_features, reference_resnet_features)
                    mobilenet_cosine_dist = cosine_distance(mobilenet_features, reference_mobilenet_features)
                    
                    # Calculate composite distance
                    composite_dist = composite_resnet_mobilenet_distance(resnet_cosine_dist, mobilenet_cosine_dist)
                    similarity_score = composite_dist
                    match = composite_dist < composite_threshold
                    if args.debug and frame_count % 30 == 0:
                        print(f"🐛 Composite: ResNet_conf={resnet_feat_conf:.3f}, MobileNet_conf={mobilenet_feat_conf:.3f}")
                        print(f"🐛 Composite Distance: {composite_dist:.4f} (ResNet: {resnet_cosine_dist:.4f}, MobileNet: {mobilenet_cosine_dist:.4f}, threshold: {composite_threshold})")
                elif resnet_valid:
                    # Only ResNet is confident - use ResNet only
                    resnet_cosine_dist = cosine_distance(resnet_features, reference_resnet_features)
                    similarity_score = resnet_cosine_dist
                    match = resnet_cosine_dist < cosine_threshold
                    if args.debug and frame_count % 30 == 0:
                        print(f"🐛 Composite (ResNet only): Distance={resnet_cosine_dist:.4f}, MobileNet confidence too low ({mobilenet_feat_conf:.3f})")
                elif mobilenet_valid:
                    # Only MobileNet is confident - use MobileNet only
                    mobilenet_cosine_dist = cosine_distance(mobilenet_features, reference_mobilenet_features)
                    similarity_score = mobilenet_cosine_dist
                    match = mobilenet_cosine_dist < mobilenet_threshold
                    if args.debug and frame_count % 30 == 0:
                        print(f"🐛 Composite (MobileNet only): Distance={mobilenet_cosine_dist:.4f}, ResNet confidence too low ({resnet_feat_conf:.3f})")
                else:
                    # Neither feature is confident enough - skip matching
                    similarity_score = 1.0  # Max distance (no match)
                    match = False
                    if args.debug and frame_count % 30 == 0:
                        print(f"🐛 Composite: Both features below confidence threshold (ResNet: {resnet_feat_conf:.3f}, MobileNet: {mobilenet_feat_conf:.3f}) - skipping match")
            # elif args.feature == "osnet":
            #     # OSNet temporarily disabled due to torchreid compatibility issues
            #     pass
            # elif args.feature == "composite":
            #     # Composite temporarily disabled due to torchreid compatibility issues
            #     pass

            if args.anpr:
                plates = lp_detector(crop_img, verbose=False, conf=lp_confidence)
                for plate in plates[0].boxes.data.tolist():
                    lpx1, lpy1, lpx2, lpy2, lpscore, _ = plate
                    lp_crop = crop_img[int(lpy1):int(lpy2), int(lpx1):int(lpx2)]
                    ocr_result = ocr.ocr(lp_crop, cls=False, det=False)
                    if ocr_result and ocr_result[0]:
                        plate_text = ocr_result[0][0][0]  # Extracting text from the first result
                        if plate_text == args.anpr:
                            match = True
                            plate_title = plate_text + " - matched"
                            # Calculate the width and height of the text box
                            (text_width, text_height) = cv2.getTextSize(plate_title, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)[0]

                            # Set the text start position
                            text_start_x = int(x1)
                            text_start_y = int(y1) + 30

                            # Draw the filled rectangle
                            cv2.rectangle(annotated_frame, (text_start_x, text_start_y - text_height), (text_start_x + text_width, text_start_y), (255, 255, 255), cv2.FILLED)

                            # Put the text on the rectangle
                            cv2.putText(annotated_frame, plate_title, (text_start_x, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)

            end_time = default_timer()
            current_feature_time = end_time - start_time
            processing_times.append(current_feature_time)

            if match:
                matches_found += 1
                # Update progress bar description with match info
                pbar.set_description(f"Processing frames (Matches: {matches_found})")
                
                # Draw a green rectangle around matched vehicles
                (text_width, text_height) = cv2.getTextSize(f'{args.feature} matched', cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)[0]
                text_start_x = int(x1)
                text_start_y = int(y2)
                cv2.rectangle(annotated_frame, (text_start_x, text_start_y - text_height), (text_start_x + text_width, text_start_y), (255, 255, 255), cv2.FILLED)
                # Put the text on the rectangle
                cv2.putText(annotated_frame, f'{args.feature} matched', (text_start_x, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 4)
                # draw green rectangle
                cv2.rectangle(annotated_frame, (x1, y1-10), (x2, y2), (0, 255, 0), 3)  # Use xyxy format

                crop_name = f"frame_{frame_count}_ID_{results[0].boxes.id[i]}_{args.feature}.jpg"
                crop_path = os.path.join(save_dir, crop_name)
                cv2.imwrite(crop_path, crop_img)
    else:
        annotated_frame = frame

    # Write frame to output video if enabled
    if video_writer is not None:
        video_writer.write(annotated_frame)

pbar.close()

# Clean up resources
cap.release()
if video_writer is not None:
    video_writer.release()

# Calculate results summary
total_processing_time = sum(processing_times)
avg_processing_time = total_processing_time / len(processing_times) if processing_times else 0
match_rate = (matches_found / total_detections * 100) if total_detections > 0 else 0

results_summary = {
    "total_frames": frame_count,
    "total_detections": total_detections,
    "matches_found": matches_found,
    "match_rate_percentage": round(match_rate, 2),
    "avg_processing_time_per_vehicle": round(avg_processing_time, 4),
    "total_processing_time": round(total_processing_time, 3),
    "reference_feature_time": round(reference_feature_time, 3)
}

print()
print("📊 RESULTS SUMMARY")
print("=" * 50)
print(f"🎬 Total Frames Processed: {frame_count}")
print(f"🚗 Total Vehicle Detections: {total_detections}")
print(f"✅ Matches Found: {matches_found}")
print(f"📈 Match Rate: {match_rate:.2f}%")
print(f"⏱️  Avg Processing Time/Vehicle: {avg_processing_time:.4f}s")
print(f"⏱️  Reference Feature Extraction: {reference_feature_time:.3f}s")
print(f"⏱️  Total Processing Time: {total_processing_time:.3f}s")
print()

# Save run information
info_path = save_run_info(exp_dir, args, results_summary)
print(f"💾 Run information saved to: {info_path}")

if args.save_video and video_writer is not None:
    print(f"🎬 Output video saved to: {output_video_path}")

print(f"🖼️  Matched vehicle images saved to: {save_dir}")
print()
print("✅ PROCESSING COMPLETE!")
print("=" * 80)