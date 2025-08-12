"""
@brief: Advanced Vehicle Re-identification System with Database Storage
@project: AI Based Smart Parking Management System - Enhanced Re-identification Module
@authors: Mohammad Saad Iqbal, Rafay Abrar
@organization: National Institute of Electronics (NIE)
@description: Two-stage vehicle re-identification system that processes reference video to build
             a vehicle database, then processes target video for re-identification with 
             NEW/RE-ID tagging and comprehensive similarity scoring.
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
from feature_extraction import *
from similarity import *
from database import VehicleDatabase, aggregate_features
# Optional: PaddleOCR is not required for this script's flow. Guard import to avoid env issues.
try:
    from paddleocr import PaddleOCR  # noqa: F401
    HAVE_PADDLE_OCR = True
except Exception:
    HAVE_PADDLE_OCR = False
from tqdm import tqdm
import sys
import numpy as np
from collections import defaultdict

# Default video file paths (set these to enable fallback when CLI is empty)
DEFAULT_REF_VID_A = "reference_vehicles/ref_v12.mp4"   # Reference video default
DEFAULT_TGT_VID_B = "videos/v12.mp4"    # Target video default

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--reference_video", default=None, help="path to reference video (falls back to DEFAULT_REF_VID_A)")
parser.add_argument("-t", "--target_video", default=None, help="path to target video for re-identification (falls back to DEFAULT_TGT_VID_B)")
parser.add_argument("-f", "--feature", default="sift", choices=["hog", "histogram", "sift", "resnet", "mobilenet", "composite"],
                    help="Feature type to use for comparison")
parser.add_argument("--threshold", type=float, default=None, help="similarity threshold (feature-specific)")
parser.add_argument("-c", "--confidence", type=float, default=None, help="detection confidence threshold (0.0-1.0)")
parser.add_argument("--save-video", action="store_true", help="save output video (default: True)")
parser.add_argument("--no-save-video", dest="save_video", action="store_false", help="don't save output video")
parser.add_argument("--debug", action="store_true", help="print similarity scores for debugging")
parser.set_defaults(save_video=True)
args = parser.parse_args()

# Resolve inputs with smart fallback to defaults
def resolve_video_sources():
    ref = args.reference_video or DEFAULT_REF_VID_A
    tgt = args.target_video or DEFAULT_TGT_VID_B

    # Normalize paths
    ref = os.path.normpath(ref) if ref else ref
    tgt = os.path.normpath(tgt) if tgt else tgt

    missing = []
    if not ref or not os.path.exists(ref):
        missing.append(f"Reference video not found: {ref}")
    if not tgt or not os.path.exists(tgt):
        missing.append(f"Target video not found: {tgt}")

    if missing:
        print("❌ Input error:")
        for m in missing:
            print(f"   • {m}")
        print("💡 Set defaults in code (DEFAULT_REF_VID_A / DEFAULT_TGT_VID_B) or pass --reference_video / --target_video")
        sys.exit(1)

    # Apply resolved sources
    args.reference_video = ref
    args.target_video = tgt

    print("✅ SMART SOURCE CONFIGURATION:")
    print(f"   • Reference: {args.reference_video}")
    print(f"   • Target: {args.target_video}")

resolve_video_sources()

def create_runs_directory():
    """Create runs_v2 directory with incremental experiment naming"""
    runs_dir = "runs_v2"
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
    print("🚗 ADVANCED VEHICLE RE-IDENTIFICATION SYSTEM V2")
    print("    AI Based Smart Parking Management System")
    print("    Two-Stage Processing with Database Storage")
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
    print(f"🎬 Reference Video: {args.reference_video}")
    print(f"🎬 Target Video: {args.target_video}")
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
    
    print(f"💾 Save Video: {'Yes' if args.save_video else 'No'}")
    if args.debug:
        print(f"🐛 Debug Mode: Enabled")
    print()

def process_reference_video(video_path, feature_method, db, exp_dir):
    """Process reference video and store vehicle features in database"""
    print("🎬 PROCESSING REFERENCE VIDEO")
    print("-" * 40)
    
    # Initialize models
    print("📦 Loading YOLOv8 model...")
    model = YOLO("models/yolov8n.pt")
    
    # Initialize feature extraction models
    feature_models = initialize_feature_models(feature_method)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open reference video file!")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Reference Video Properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Add video to database
    video_id = db.add_video(video_path, 'reference', feature_method, total_frames, 0)
    
    # Setup video writers for both reference and target videos
    ref_video_writer = None
    target_video_writer = None
    
    if args.save_video:
        # Reference video output
        ref_output_path = os.path.join(exp_dir, f"reference_output_{feature_method}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        ref_video_writer = cv2.VideoWriter(ref_output_path, fourcc, fps, (width, height))
        print(f"💾 Reference video output will be saved to: {ref_output_path}")
    
    # Track vehicles across frames
    vehicle_tracks = defaultdict(list)  # track_id -> list of (frame, bbox, features)
    frame_count = 0
    total_detections = 0
    
    class_list = [2]  # Only cars
    
    # Progress bar
    pbar = tqdm(total=total_frames, desc="Processing reference video", unit="frame")
    
    # Create mapping from track_id to vehicle_id for consistent annotation and database storage
    track_to_vehicle_mapping = {}
    vehicle_counter = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        pbar.update(1)
        
        results = model.track(frame, show=False, verbose=False, conf=yolo_confidence, classes=class_list, persist=True)
        annotated_frame = frame.copy()
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            total_detections += len(boxes)
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                crop_img = frame[y1:y2, x1:x2]
                track_id = track_ids[i]
                
                # Extract features
                features, confidence = extract_features_with_confidence(crop_img, feature_method, feature_models)
                
                if features is not None:
                    vehicle_tracks[track_id].append({
                        'frame': frame_count,
                        'bbox': [x1, y1, x2, y2],
                        'features': features,
                        'confidence': confidence
                    })
                
                # Create vehicle_id mapping when vehicle has enough appearances
                if len(vehicle_tracks[track_id]) >= 3 and track_id not in track_to_vehicle_mapping:
                    track_to_vehicle_mapping[track_id] = f"A{vehicle_counter}"
                    vehicle_counter += 1
                
                # Draw annotations for reference video using consistent vehicle_id
                if track_id in track_to_vehicle_mapping:
                    ref_vehicle_id = track_to_vehicle_mapping[track_id]
                    draw_reference_annotations(annotated_frame, x1, y1, x2, y2, ref_vehicle_id, 
                                             confidence, feature_method.upper())
        
        # Write frame to reference video output
        if ref_video_writer is not None:
            ref_video_writer.write(annotated_frame)
    
    pbar.close()
    cap.release()
    if ref_video_writer is not None:
        ref_video_writer.release()
    
    # Process and store vehicle tracks
    print("💾 Storing vehicle tracks and features...")
    stored_vehicles = 0
    
    for track_id, appearances in vehicle_tracks.items():
        if len(appearances) >= 3:  # Only store vehicles that appear in at least 3 frames
            vehicle_id = track_to_vehicle_mapping[track_id]  # Use the same vehicle_id from annotation
            
            first_frame = min(app['frame'] for app in appearances)
            last_frame = max(app['frame'] for app in appearances)
            total_appearances = len(appearances)
            
            # Prepare bbox data
            bbox_data = [{'frame': app['frame'], 'bbox': app['bbox']} for app in appearances]
            
            # Add vehicle to database
            vehicle_table_id = db.add_vehicle(video_id, vehicle_id, track_id, first_frame, last_frame, total_appearances, bbox_data)
            
            # Store individual features
            features_list = []
            confidences = []
            
            for app in appearances:
                db.add_vehicle_feature(vehicle_table_id, app['frame'], app['features'], app['confidence'], feature_method)
                features_list.append(app['features'])
                confidences.append(app['confidence'])
            
            # Calculate and store aggregated features
            aggregated_features = aggregate_features(features_list, method='mean')
            avg_confidence = np.mean(confidences)
            
            db.add_aggregated_features(vehicle_table_id, aggregated_features, feature_method, len(features_list), avg_confidence)
            stored_vehicles += 1
    
    print(f"✅ Reference video processed: {stored_vehicles} vehicles stored in database")
    return True

def process_target_video(video_path, feature_method, db, exp_dir):
    """Process target video and perform smart re-identification with unique ID assignment"""
    print("🎬 PROCESSING TARGET VIDEO FOR SMART RE-IDENTIFICATION")
    print("-" * 40)
    
    # Initialize models
    model = YOLO("models/yolov8n.pt")
    feature_models = initialize_feature_models(feature_method)
    
    # Get reference vehicles from database
    reference_vehicles = db.get_reference_vehicles(feature_method)
    print(f"📚 Loaded {len(reference_vehicles)} reference vehicles from database")
    
    if not reference_vehicles:
        print("❌ No reference vehicles found in database!")
        return False
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open target video file!")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Target Video Properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Add target video to database
    target_video_id = db.add_video(video_path, 'target', feature_method, total_frames, 0)
    
    # Setup video writer if saving video
    video_writer = None
    if args.save_video:
        output_video_path = os.path.join(exp_dir, f"target_output_{feature_method}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"💾 Target video output will be saved to: {output_video_path}")
    
    # Track target vehicles and collect all data first
    target_tracks = defaultdict(list)
    frame_count = 0
    class_list = [2]  # Only cars
    
    # Progress bar for feature extraction
    pbar = tqdm(total=total_frames, desc="Extracting features from target video", unit="frame")
    
    # PHASE 1: Extract all features from target video
    print("🔍 Phase 1: Extracting features from all target vehicles...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        pbar.update(1)
        
        results = model.track(frame, show=False, verbose=False, conf=yolo_confidence, classes=class_list, persist=True)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                crop_img = frame[y1:y2, x1:x2]
                track_id = track_ids[i]
                
                # Extract features
                features, confidence = extract_features_with_confidence(crop_img, feature_method, feature_models)
                
                if features is not None:
                    target_tracks[track_id].append({
                        'frame': frame_count,
                        'bbox': [x1, y1, x2, y2],
                        'features': features,
                        'confidence': confidence
                    })
    
    pbar.close()
    cap.release()
    
    # PHASE 2: Smart Re-identification with unique ID assignment
    print("🧠 Phase 2: Performing smart re-identification with unique ID assignment...")
    
    # Prepare target vehicles with aggregated features
    target_vehicle_candidates = []
    for track_id, appearances in target_tracks.items():
        if len(appearances) >= 3:  # Minimum appearances requirement
            features_list = [app['features'] for app in appearances]
            confidences = [app['confidence'] for app in appearances]
            target_aggregated = aggregate_features(features_list, method='mean')
            avg_confidence = np.mean(confidences)
            
            target_vehicle_candidates.append({
                'track_id': track_id,
                'appearances': appearances,
                'aggregated_features': target_aggregated,
                'avg_confidence': avg_confidence,
                'features_list': features_list,
                'confidences': confidences
            })
    
    print(f"🎯 Found {len(target_vehicle_candidates)} target vehicle candidates")
    
    # Smart assignment: Each reference ID can only be assigned to ONE target vehicle
    id_assignments = perform_smart_reid_assignment(target_vehicle_candidates, reference_vehicles, feature_method)
    
    # Debug output for assignments
    if args.debug:
        print(f"\n🔍 DEBUG: Smart assignments made:")
        for track_id, assignment in id_assignments.items():
            print(f"   Track {track_id} → {assignment['reference_id']} (similarity: {assignment['similarity']:.3f})")
    
    # PHASE 3: Generate annotated video with smart assignments
    print("🎨 Phase 3: Generating annotated output video...")
    
    # Prepare candidate bounding boxes by frame for IoU matching
    candidate_bboxes_by_frame = {}
    candidate_assignments = {}
    new_vehicle_counter = 0
    for candidate in target_vehicle_candidates:
        track_id = candidate['track_id']
        assignment = id_assignments.get(track_id)
        if assignment:
            candidate_assignments[track_id] = {
                'vehicle_id': assignment['reference_id'],
                'tag': 'RE-ID',
                'similarity': assignment['similarity'],
                'is_match': True
            }
        else:
            candidate_assignments[track_id] = {
                'vehicle_id': f"B{new_vehicle_counter}",
                'tag': 'NEW',
                'similarity': 0.0,
                'is_match': False
            }
            new_vehicle_counter += 1
        # Store bboxes by frame for this candidate
        for app in candidate['appearances']:
            frame_num = app['frame']
            bbox = app['bbox']
            if frame_num not in candidate_bboxes_by_frame:
                candidate_bboxes_by_frame[frame_num] = []
            candidate_bboxes_by_frame[frame_num].append({
                'track_id': track_id,
                'bbox': bbox
            })

    print("🎨 Phase 3: Generating annotated output video...")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Generating annotated video", unit="frame")

    def compute_iou(boxA, boxB):
        # box = [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    IOU_THRESHOLD = 0.3

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        pbar.update(1)
        results = model.track(frame, show=False, verbose=False, conf=yolo_confidence, classes=class_list, persist=True)
        annotated_frame = frame.copy()
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                det_box = [x1, y1, x2, y2]
                # Find best matching candidate by IoU for this frame
                best_iou = 0
                best_track_id = None
                if frame_count in candidate_bboxes_by_frame:
                    for cand in candidate_bboxes_by_frame[frame_count]:
                        cand_box = cand['bbox']
                        iou = compute_iou(det_box, cand_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_track_id = cand['track_id']
                if best_iou >= IOU_THRESHOLD and best_track_id is not None:
                    final_assignment = candidate_assignments[best_track_id]
                    vehicle_id = final_assignment['vehicle_id']
                    tag = final_assignment['tag']
                    similarity = final_assignment['similarity']
                    is_match = final_assignment['is_match']
                    if args.debug:
                        print(f"[DEBUG] Frame {frame_count}: IoU {best_iou:.2f} for track {best_track_id} → {vehicle_id}")
                    draw_annotations(annotated_frame, x1, y1, x2, y2, vehicle_id, tag, similarity, feature_method.upper(), is_match)
        if video_writer is not None:
            video_writer.write(annotated_frame)
    pbar.close()
    cap.release()
    if video_writer is not None:
        video_writer.release()
    
    # PHASE 4: Store results in database
    print("💾 Phase 4: Storing smart re-identification results in database...")
    
    target_vehicle_counter = 0
    reid_results = []
    for candidate in target_vehicle_candidates:
        track_id = candidate['track_id']
        appearances = candidate['appearances']
        # Use the same final ID assignment from the IoU-based mapping
        final_assignment = candidate_assignments[track_id]
        final_vehicle_id = final_assignment['vehicle_id']
        first_frame = min(app['frame'] for app in appearances)
        last_frame = max(app['frame'] for app in appearances)
        total_appearances = len(appearances)
        bbox_data = [{'frame': app['frame'], 'bbox': app['bbox']} for app in appearances]
        # Add target vehicle to database
        target_table_id = db.add_vehicle(target_video_id, final_vehicle_id, track_id, 
                                       first_frame, last_frame, total_appearances, bbox_data)
        # Store features
        for app in appearances:
            db.add_vehicle_feature(target_table_id, app['frame'], app['features'], app['confidence'], feature_method)
        # Store aggregated features
        db.add_aggregated_features(target_table_id, candidate['aggregated_features'], 
                                 feature_method, len(candidate['features_list']), candidate['avg_confidence'])
        # Store re-identification results for all reference vehicles
        best_match = None
        best_similarity = float('inf')
        for ref_vehicle in reference_vehicles:
            similarity = calculate_similarity(candidate['aggregated_features'], ref_vehicle['features'], feature_method)
            is_match = (final_assignment['tag'] == 'RE-ID' and final_assignment['vehicle_id'] == ref_vehicle['vehicle_id'])
            db.add_reid_result(target_table_id, ref_vehicle['table_id'], similarity, is_match, feature_method)
            if similarity < best_similarity:
                best_similarity = similarity
                best_match = ref_vehicle
        reid_results.append({
            'target_id': final_vehicle_id,
            'best_match': best_match['vehicle_id'] if best_match else None,
            'similarity': best_similarity,
            'is_match': final_assignment['tag'] == 'RE-ID',
            'assignment_type': final_assignment['tag']
        })
    
    # Print smart assignment summary
    print("\n🎯 SMART RE-IDENTIFICATION SUMMARY:")
    print("-" * 50)
    reidentified = sum(1 for r in reid_results if r['assignment_type'] == 'RE-ID')
    new_vehicles = sum(1 for r in reid_results if r['assignment_type'] == 'NEW')
    print(f"✅ Re-identified vehicles: {reidentified}")
    print(f"🆕 New vehicles: {new_vehicles}")
    print(f"📊 Total target vehicles: {len(reid_results)}")
    
    for result in reid_results:
        if result['assignment_type'] == 'RE-ID':
            print(f"   🔄 {result['target_id']} ← Match with reference vehicle (similarity: {result['similarity']:.3f})")
        else:
            print(f"   🆕 {result['target_id']} ← New vehicle (no match found)")
    
    print(f"\n✅ Smart target video processing completed: {len(reid_results)} vehicles analyzed")
    return True

def initialize_feature_models(feature_method):
    """Initialize feature extraction models based on method"""
    models = {}
    
    if feature_method == "resnet":
        models['resnet'] = models_torch.resnet50(pretrained=True)
        models['resnet'].eval()
    elif feature_method == "mobilenet":
        models['mobilenet'] = models_torch.mobilenet_v2(pretrained=True)
        models['mobilenet'].eval()
    elif feature_method == "composite":
        models['resnet'] = models_torch.resnet50(pretrained=True)
        models['resnet'].eval()
        models['mobilenet'] = models_torch.mobilenet_v2(pretrained=True)
        models['mobilenet'].eval()
    elif feature_method == "sift":
        models['sift'] = cv2.SIFT_create()
    
    return models

def extract_features_with_confidence(image, feature_method, feature_models):
    """Extract features with confidence score"""
    try:
        if feature_method == "hog":
            features = compute_hog_features(image)
            confidence = 0.8  # Default confidence for HOG
            return features, confidence
        
        elif feature_method == "histogram":
            features = compute_color_histogram(image)
            confidence = 0.8  # Default confidence for histogram
            return features, confidence
        
        elif feature_method == "sift":
            sift = feature_models['sift']
            keypoints, descriptors = sift.detectAndCompute(image, None)
            if descriptors is not None:
                # Use average descriptor as feature representation
                features = np.mean(descriptors, axis=0)
                confidence = min(1.0, len(keypoints) / 100.0)  # Confidence based on keypoint count
                return features, confidence
            return None, 0.0
        
        elif feature_method == "resnet":
            features, confidence = extract_dnn_features(feature_models['resnet'], image, resnet_confidence)
            return features, confidence
        
        elif feature_method == "mobilenet":
            features, confidence = extract_dnn_features(feature_models['mobilenet'], image, mobilenet_confidence)
            return features, confidence
        
        elif feature_method == "composite":
            resnet_features, resnet_conf = extract_dnn_features(feature_models['resnet'], image, resnet_confidence)
            mobilenet_features, mobilenet_conf = extract_dnn_features(feature_models['mobilenet'], image, mobilenet_confidence)
            
            # Combine features
            if resnet_conf >= resnet_confidence and mobilenet_conf >= mobilenet_confidence:
                combined_features = np.concatenate([resnet_features, mobilenet_features])
                avg_confidence = (resnet_conf + mobilenet_conf) / 2
                return combined_features, avg_confidence
            elif resnet_conf >= resnet_confidence:
                return resnet_features, resnet_conf
            elif mobilenet_conf >= mobilenet_confidence:
                return mobilenet_features, mobilenet_conf
            else:
                return None, 0.0
    
    except Exception as e:
        if args.debug:
            print(f"⚠️ Feature extraction error: {e}")
        return None, 0.0
    
    return None, 0.0

def calculate_similarity(features1, features2, feature_method):
    """Calculate similarity between two feature vectors"""
    if feature_method == "hog":
        return l2_distance(features1, features2)
    elif feature_method == "histogram":
        return 1.0 - histogram_distance(features1, features2)  # Convert correlation to distance
    elif feature_method in ["resnet", "mobilenet", "composite"]:
        return cosine_distance(features1, features2)
    elif feature_method == "sift":
        return l2_distance(features1, features2)  # Using L2 for aggregated SIFT features
    
    return float('inf')

def is_similarity_match(similarity_score, feature_method):
    """Determine if similarity score indicates a match"""
    if feature_method == "hog":
        return similarity_score < hog_threshold
    elif feature_method == "histogram":
        return similarity_score < (1.0 - histogram_threshold)  # Adjusted for distance
    elif feature_method == "resnet":
        return similarity_score < cosine_threshold
    elif feature_method == "mobilenet":
        return similarity_score < mobilenet_threshold
    elif feature_method == "composite":
        return similarity_score < composite_threshold
    elif feature_method == "sift":
        return similarity_score < sift_threshold
    
    return False

def perform_smart_reid_assignment(target_candidates, reference_vehicles, feature_method):
    """
    🧠 SMART RE-IDENTIFICATION ASSIGNMENT ALGORITHM
    
    This function implements intelligent vehicle re-identification with unique ID constraints:
    1. Each reference vehicle ID (A0, A1, A2, etc.) can only be assigned to ONE target vehicle
    2. Among multiple target candidates for the same reference ID, only the MOST similar one gets assigned
    3. All assignments must meet the similarity threshold requirements
    
    Args:
        target_candidates: List of target vehicle candidates with aggregated features
        reference_vehicles: List of reference vehicles from database
        feature_method: Feature extraction method being used
    
    Returns:
        dict: Mapping of track_id -> {reference_id, similarity, confidence} for assigned vehicles
    """
    
    print("🧠 Performing smart re-identification assignment...")
    
    # Step 1: Calculate all possible matches above threshold
    potential_matches = []
    
    for target_idx, target in enumerate(target_candidates):
        for ref_vehicle in reference_vehicles:
            similarity = calculate_similarity(target['aggregated_features'], ref_vehicle['features'], feature_method)
            
            # Check if this similarity meets the threshold
            if is_similarity_match(similarity, feature_method):
                potential_matches.append({
                    'target_idx': target_idx,
                    'target_track_id': target['track_id'],
                    'reference_id': ref_vehicle['vehicle_id'],
                    'reference_table_id': ref_vehicle['table_id'],
                    'similarity': similarity,
                    'target_confidence': target['avg_confidence']
                })
    
    print(f"   Found {len(potential_matches)} potential matches above threshold")
    
    # Step 2: Sort matches by similarity (best matches first)
    potential_matches.sort(key=lambda x: x['similarity'])
    
    # Step 3: Smart assignment - each reference ID can only be assigned once
    assigned_reference_ids = set()
    assigned_target_tracks = set()
    final_assignments = {}
    
    for match in potential_matches:
        ref_id = match['reference_id']
        target_track = match['target_track_id']
        
        # Skip if this reference ID or target track is already assigned
        if ref_id in assigned_reference_ids or target_track in assigned_target_tracks:
            continue
        
        # Assign this match
        final_assignments[target_track] = {
            'reference_id': ref_id,
            'similarity': match['similarity'],
            'confidence': match['target_confidence'],
            'reference_table_id': match['reference_table_id']
        }
        
        assigned_reference_ids.add(ref_id)
        assigned_target_tracks.add(target_track)
        
        if args.debug:
            print(f"   ✅ Assigned {ref_id} to target track {target_track} (similarity: {match['similarity']:.3f})")
    
    print(f"   🎯 Smart assignment completed: {len(final_assignments)} unique assignments made")
    
    # Step 4: Report assignment statistics
    total_references = len(reference_vehicles)
    total_targets = len(target_candidates)
    reidentified = len(final_assignments)
    
    print(f"   📊 Assignment Statistics:")
    print(f"      • Reference vehicles available: {total_references}")
    print(f"      • Target vehicles found: {total_targets}")
    print(f"      • Successfully re-identified: {reidentified}")
    print(f"      • New vehicles: {total_targets - reidentified}")
    
    return final_assignments

def draw_reference_annotations(frame, x1, y1, x2, y2, vehicle_id, confidence, model_name):
    """Draw annotations on the reference video frame"""
    # Colors for reference video (blue theme)
    ref_color = (255, 100, 0)  # Blue for reference vehicles
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), ref_color, 2)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Upper Left Corner: Vehicle ID
    id_text = vehicle_id
    (id_w, id_h), _ = cv2.getTextSize(id_text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1-id_h-5), (x1+id_w+5, y1), bg_color, -1)
    cv2.putText(frame, id_text, (x1+2, y1-5), font, font_scale, text_color, thickness)
    
    # Upper Right Corner: Confidence
    conf_text = f"Conf: {confidence:.3f}"
    (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale, thickness)
    cv2.rectangle(frame, (x2-conf_w-5, y1-conf_h-5), (x2, y1), bg_color, -1)
    cv2.putText(frame, conf_text, (x2-conf_w-2, y1-5), font, font_scale, text_color, thickness)
    
    # Bottom Left Corner: Model Name
    model_text = f"REF-{model_name}"
    (model_w, model_h), _ = cv2.getTextSize(model_text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y2), (x1+model_w+5, y2+model_h+5), bg_color, -1)
    cv2.putText(frame, model_text, (x1+2, y2+model_h), font, font_scale, text_color, thickness)

def draw_annotations(frame, x1, y1, x2, y2, vehicle_id, tag, similarity, model_name, is_match):
    """Draw annotations on the frame"""
    # Colors based on tag type
    if tag == "RE-ID":
        match_color = (0, 255, 0)  # Green for RE-ID
    elif tag == "NEW":
        match_color = (0, 0, 255)  # Red for NEW
    elif tag == "PROCESSING":
        match_color = (0, 165, 255)  # Orange for processing
    else:
        match_color = (128, 128, 128)  # Gray for unknown
    
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), match_color, 2)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Upper Left Corner: Vehicle ID
    id_text = vehicle_id
    (id_w, id_h), _ = cv2.getTextSize(id_text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1-id_h-5), (x1+id_w+5, y1), bg_color, -1)
    cv2.putText(frame, id_text, (x1+2, y1-5), font, font_scale, text_color, thickness)
    
    # Upper Right Corner: Tag and Similarity (only show similarity if not processing)
    if tag == "PROCESSING":
        tag_text = tag
    else:
        tag_text = f"{tag}: {similarity:.3f}"
    
    (tag_w, tag_h), _ = cv2.getTextSize(tag_text, font, font_scale, thickness)
    cv2.rectangle(frame, (x2-tag_w-5, y1-tag_h-5), (x2, y1), bg_color, -1)
    cv2.putText(frame, tag_text, (x2-tag_w-2, y1-5), font, font_scale, text_color, thickness)
    
    # Bottom Left Corner: Model Name
    model_text = model_name
    (model_w, model_h), _ = cv2.getTextSize(model_text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y2), (x1+model_w+5, y2+model_h+5), bg_color, -1)
    cv2.putText(frame, model_text, (x1+2, y2+model_h), font, font_scale, text_color, thickness)

# ===============================================================================
# SIMILARITY THRESHOLDS - Adjust these values to control matching sensitivity
# Lower values = more strict matching (fewer matches but higher confidence)
# Higher values = more lenient matching (more matches but lower confidence)
# ===============================================================================

default_hog_threshold = 0.5          # HOG L2 distance threshold (lower = stricter)
default_histogram_threshold = 0.85   # Color histogram correlation threshold (higher = stricter)
default_resnet_threshold = 0.375     # ResNet cosine distance threshold (lower = stricter)
default_mobilenet_threshold = 0.3  # MobileNetV2 cosine distance threshold (lower = stricter)
default_sift_threshold = 0.5         # SIFT aggregated features L2 threshold (lower = stricter)
default_composite_threshold = 0.375  # Composite method distance threshold (lower = stricter)

# ===============================================================================
# CONFIDENCE THRESHOLDS - Adjust these values to control feature quality
# Higher values = require better quality features (more reliable but may skip some)
# Lower values = accept lower quality features (more inclusive but less reliable)
# ===============================================================================

default_yolo_confidence = 0.6        # YOLO vehicle detection confidence (0.0-1.0)
default_resnet_confidence = 0.8      # ResNet feature extraction confidence (0.0-1.0)
default_mobilenet_confidence = 0.8   # MobileNetV2 feature extraction confidence (0.0-1.0)

# Note: For composite method, both ResNet and MobileNet confidences are used
# If both models meet confidence threshold, features are combined
# If only one meets threshold, that model's features are used
# If neither meets threshold, the vehicle detection is skipped

# Use command-line threshold if provided, otherwise use defaults
hog_threshold = args.threshold if args.threshold is not None else default_hog_threshold
histogram_threshold = args.threshold if args.threshold is not None else default_histogram_threshold
cosine_threshold = args.threshold if args.threshold is not None else default_resnet_threshold
mobilenet_threshold = args.threshold if args.threshold is not None else default_mobilenet_threshold
sift_threshold = args.threshold if args.threshold is not None else default_sift_threshold
composite_threshold = args.threshold if args.threshold is not None else default_composite_threshold

yolo_confidence = args.confidence if args.confidence is not None else default_yolo_confidence
resnet_confidence = args.confidence if args.confidence is not None else default_resnet_confidence
mobilenet_confidence = args.confidence if args.confidence is not None else default_mobilenet_confidence

# Import torch models with different name to avoid conflict
import torchvision.models as models_torch

def main():
    """Main function"""
    # Create experiment directory
    exp_dir, exp_name = create_runs_directory()
    
    # Print header and configuration
    print_header()
    print_config(args, exp_name, exp_dir)
    
    # Initialize database - always create fresh database for each run
    db_path = os.path.join(exp_dir, "vehicle_reid.db")
    print(f"🗃️  Creating fresh database: {db_path}")
    
    # Remove existing database if it exists (ensure clean slate)
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = VehicleDatabase(db_path)
    
    try:
        # Stage 1: Process reference video
        print("=" * 80)
        print("STAGE 1: BUILDING REFERENCE DATABASE")
        print("=" * 80)
        
        success = process_reference_video(args.reference_video, args.feature, db, exp_dir)
        if not success:
            print("❌ Failed to process reference video!")
            return
        
        # Stage 2: Process target video and perform re-identification
        print("\n" + "=" * 80)
        print("STAGE 2: TARGET VIDEO RE-IDENTIFICATION")
        print("=" * 80)
        
        success = process_target_video(args.target_video, args.feature, db, exp_dir)
        if not success:
            print("❌ Failed to process target video!")
            return
        
        print("\n" + "=" * 80)
        print("✅ PROCESSING COMPLETE!")
        print(f"📁 Results saved in: {exp_dir}")
        print(f"🗃️  Database saved as: {db_path}")
        if args.save_video:
            print(f"🎬 Reference video: {exp_dir}/reference_output_{args.feature}.mp4")
            print(f"🎬 Target video: {exp_dir}/target_output_{args.feature}.mp4")
        print("=" * 80)
        
    finally:
        db.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
