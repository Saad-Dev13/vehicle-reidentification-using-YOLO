"""
@brief: Live Vehicle Re-identification System for CCTV Environment
@project: AI Based Smart Parking Management System - Live Re-identification Module
@authors: Mohammad Saad Iqbal, Rafay Abrar
@organization: National Institute of Electronics (NIE)
@description: Real-time vehicle re-identification system that processes two video feeds 
             simultaneously - Camera A (reference) and Camera B (target) - with live 
             display and real-time re-identification processing.
"""

# Fix SSL certificate issues for downloading models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import os
import argparse
import glob
import json
import threading
import time
from datetime import datetime
from timeit import default_timer
from ultralytics import YOLO
from torchvision import models
from feature_extraction import *
from similarity import *
from database import VehicleDatabase, aggregate_features
from paddleocr import PaddleOCR
import sys
import numpy as np
from collections import defaultdict, deque
import queue

# Default camera IP configurations
DEFAULT_CAMERA_IP_A = "rtsp://admin:password@192.168.1.100:554/0"  # Reference camera
DEFAULT_CAMERA_IP_B = "rtsp://admin:password@192.168.1.101:554/0"  # Target camera

# Default video file paths (set to empty string "" to use cameras instead)
DEFAULT_REF_VID_A = "videos/v17.mp4"  # Reference video file
DEFAULT_TGT_VID_B = ""  # Target video file "videos/v12.mp4"

# Camera start delays (in seconds) - useful for accurate result analysis
DEFAULT_CAMERA_A_DELAY = 0  # Reference camera/video start delay (seconds)
DEFAULT_CAMERA_B_DELAY = 180  # Target camera/video start delay (seconds)

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--reference_video", default=None, help="path to reference video (Camera A) - defaults to IP camera")
parser.add_argument("-t", "--target_video", default=None, help="path to target video (Camera B) - defaults to IP camera")
parser.add_argument("-f", "--feature", default="sift", choices=["hog", "histogram", "sift", "resnet", "mobilenet", "composite"],
                    help="Feature type to use for comparison")
parser.add_argument("--threshold", type=float, default=None, help="similarity threshold (feature-specific)")
parser.add_argument("-c", "--confidence", type=float, default=None, help="detection confidence threshold (0.0-1.0)")
parser.add_argument("--save-video", action="store_true", help="save output video (default: False for live)")
parser.add_argument("--no-save-video", dest="save_video", action="store_false", help="don't save output video")
parser.add_argument("--debug", action="store_true", help="print similarity scores for debugging")
parser.add_argument("--display-scale", type=float, default=0.5, help="scale factor for display windows (0.1-1.0)")
parser.add_argument("--fps-limit", type=int, default=15, help="maximum FPS for processing (default: 15 for speed)")
parser.add_argument("--buffer-size", type=int, default=8, help="reference vehicle buffer size (default: 8)")
parser.add_argument("--frame-skip", type=int, default=2, help="process every Nth frame for speed (default: 2)")
parser.add_argument("--camera-ip-a", type=str, default=DEFAULT_CAMERA_IP_A, help="IP camera URL for reference feed")
parser.add_argument("--camera-ip-b", type=str, default=DEFAULT_CAMERA_IP_B, help="IP camera URL for target feed")
parser.add_argument("--use-cameras", action="store_true", help="force use of IP cameras instead of video files")
parser.add_argument("--camera-a-delay", type=int, default=DEFAULT_CAMERA_A_DELAY, help=f"start delay for Camera A in seconds (default: {DEFAULT_CAMERA_A_DELAY})")
parser.add_argument("--camera-b-delay", type=int, default=DEFAULT_CAMERA_B_DELAY, help=f"start delay for Camera B in seconds (default: {DEFAULT_CAMERA_B_DELAY})")
parser.set_defaults(save_video=False)
args = parser.parse_args()

# Determine video sources (IP cameras or video files)
def get_video_sources():
    """Determine video sources with smart fallback: defaults -> command args -> cameras"""
    
    # If use-cameras is explicitly set, force IP cameras
    if args.use_cameras:
        reference_source = args.camera_ip_a
        target_source = args.camera_ip_b
        print(f"🔧 Forcing IP camera mode")
        print(f"   • Camera A (Reference): {reference_source}")
        print(f"   • Camera B (Target): {target_source}")
        return reference_source, target_source
    
    # Smart source selection with fallback logic
    reference_source = None
    target_source = None
    
    # === REFERENCE SOURCE SELECTION ===
    # 1. Check command line argument first
    if args.reference_video:
        if os.path.exists(args.reference_video):
            reference_source = args.reference_video
            print(f"🎥 Reference: Command line video ({args.reference_video})")
        elif args.reference_video.startswith('rtsp://'):
            reference_source = args.reference_video
            print(f"📹 Reference: Command line camera ({args.reference_video})")
        else:
            print(f"⚠️ Reference video not found: {args.reference_video}")
            reference_source = None
    
    # 2. If no command line arg or file not found, try default video
    if not reference_source:
        if DEFAULT_REF_VID_A and DEFAULT_REF_VID_A.strip() != "":
            if os.path.exists(DEFAULT_REF_VID_A):
                reference_source = DEFAULT_REF_VID_A
                print(f"🎥 Reference: Default video ({DEFAULT_REF_VID_A})")
            else:
                print(f"⚠️ Default reference video not found: {DEFAULT_REF_VID_A}")
    
    # 3. Final fallback to default camera
    if not reference_source:
        reference_source = args.camera_ip_a
        print(f"📹 Reference: Default IP camera ({args.camera_ip_a})")
    
    # === TARGET SOURCE SELECTION ===
    # 1. Check command line argument first
    if args.target_video:
        if os.path.exists(args.target_video):
            target_source = args.target_video
            print(f"🎥 Target: Command line video ({args.target_video})")
        elif args.target_video.startswith('rtsp://'):
            target_source = args.target_video
            print(f"📹 Target: Command line camera ({args.target_video})")
        else:
            print(f"⚠️ Target video not found: {args.target_video}")
            target_source = None
    
    # 2. If no command line arg or file not found, try default video
    if not target_source:
        if DEFAULT_TGT_VID_B and DEFAULT_TGT_VID_B.strip() != "":
            if os.path.exists(DEFAULT_TGT_VID_B):
                target_source = DEFAULT_TGT_VID_B
                print(f"🎥 Target: Default video ({DEFAULT_TGT_VID_B})")
            else:
                print(f"⚠️ Default target video not found: {DEFAULT_TGT_VID_B}")
    
    # 3. Final fallback to default camera
    if not target_source:
        target_source = args.camera_ip_b
        print(f"📹 Target: Default IP camera ({args.camera_ip_b})")
    
    # Show final configuration
    ref_type = "📹 IP Camera" if reference_source.startswith('rtsp://') else "🎥 Video File"
    target_type = "📹 IP Camera" if target_source.startswith('rtsp://') else "🎥 Video File"
    
    print(f"✅ SMART SOURCE CONFIGURATION:")
    print(f"   • Reference: {ref_type}")
    print(f"   • Target: {target_type}")
    
    return reference_source, target_source

# Get the actual video sources
REFERENCE_SOURCE, TARGET_SOURCE = get_video_sources()

# Global variables for live processing with speed optimizations
reference_vehicles_buffer = deque(maxlen=args.buffer_size)  # Rolling buffer of recent reference vehicles
target_frame_queue = queue.Queue(maxsize=5)  # Queue for target frames to process
reference_frame_queue = queue.Queue(maxsize=5)  # Queue for reference frames to process
processing_stats = {
    'reference_fps': 0,
    'target_fps': 0,
    'processing_fps': 0,
    'total_ref_vehicles': 0,
    'total_target_vehicles': 0,
    'reidentified_vehicles': 0,
    'new_vehicles': 0
}

class LiveVehicleTracker:
    """Class to manage live vehicle tracking and re-identification"""
    
    def __init__(self, feature_method, exp_dir):
        self.feature_method = feature_method
        self.exp_dir = exp_dir
        self.feature_models = initialize_feature_models(feature_method)
        
        # Initialize YOLO model with error handling and fallback
        try:
            # Try loading from models directory first
            if os.path.exists("models/yolov8n.pt"):
                self.yolo_model = YOLO("models/yolov8n.pt")
                print("✅ YOLO model loaded from models/yolov8n.pt")
            else:
                self.yolo_model = YOLO("yolov8n.pt")  # Download if not found
                print("✅ YOLO model downloaded and loaded")
                
        except Exception as e:
            print(f"⚠️ Error loading YOLO model: {e}")
            print("🔄 Trying alternative loading method...")
            try:
                # Alternative: Force download fresh model
                import ultralytics
                print(f"📦 Ultralytics version: {ultralytics.__version__}")
                self.yolo_model = YOLO("yolov8n.pt", verbose=False)
                print("✅ YOLO model loaded with alternative method")
            except Exception as e2:
                print(f"❌ All YOLO loading methods failed: {e2}")
                print("💡 Please run 'python test_yolo.py' to diagnose the issue")
                raise e2
        
        # Vehicle tracking for both cameras
        self.reference_tracks = defaultdict(list)
        self.target_tracks = defaultdict(list)
        
        # ID assignment tracking
        self.ref_track_to_vehicle = {}
        self.ref_vehicle_counter = 0
        self.target_assignments = {}
        self.new_vehicle_counter = 0
        
        # Performance tracking
        self.last_fps_time = time.time()
        self.frame_count = 0
        
        # Initialize database with persistent storage and cleaning
        db_path = os.path.join(exp_dir, "vehicle_reid.db")
        
        # Clean database before starting (ensure fresh start)
        if os.path.exists(db_path):
            print(f"🧹 Cleaning existing database: {db_path}")
            os.remove(db_path)
        
        print(f"🗃️  Creating clean database: {db_path}")
        self.db = VehicleDatabase(db_path)
    
    def cleanup_database(self):
        """Clean up database after session completion"""
        try:
            if hasattr(self, 'db') and self.db:
                db_path = self.db.db_path
                self.db.close()
                
                if os.path.exists(db_path):
                    print(f"🧹 Cleaning up database: {db_path}")
                    os.remove(db_path)
                    print("✅ Database cleaned successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean database: {e}")
    
    def update_reference_vehicle(self, track_id, frame_num, bbox, features, confidence):
        """Update reference vehicle with new appearance"""
        self.reference_tracks[track_id].append({
            'frame': frame_num,
            'bbox': bbox,
            'features': features,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Assign vehicle ID when enough appearances
        if len(self.reference_tracks[track_id]) >= 3 and track_id not in self.ref_track_to_vehicle:
            vehicle_id = f"A{self.ref_vehicle_counter}"
            self.ref_track_to_vehicle[track_id] = vehicle_id
            self.ref_vehicle_counter += 1
            
            # Add to rolling buffer
            features_list = [app['features'] for app in self.reference_tracks[track_id]]
            aggregated_features = aggregate_features(features_list, method='mean')
            avg_confidence = np.mean([app['confidence'] for app in self.reference_tracks[track_id]])
            
            reference_vehicle = {
                'vehicle_id': vehicle_id,
                'track_id': track_id,
                'features': aggregated_features,
                'confidence': avg_confidence,
                'appearances': len(self.reference_tracks[track_id]),
                'last_seen': time.time()
            }
            
            reference_vehicles_buffer.append(reference_vehicle)
            processing_stats['total_ref_vehicles'] = len(reference_vehicles_buffer)
            
            if args.debug:
                print(f"🔵 New reference vehicle: {vehicle_id} (Track {track_id})")
    
    def process_target_vehicle(self, track_id, frame_num, bbox, features, confidence):
        """Process target vehicle and perform real-time re-identification"""
        self.target_tracks[track_id].append({
            'frame': frame_num,
            'bbox': bbox,
            'features': features,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Perform re-identification when enough appearances
        if len(self.target_tracks[track_id]) >= 3 and track_id not in self.target_assignments:
            features_list = [app['features'] for app in self.target_tracks[track_id]]
            target_aggregated = aggregate_features(features_list, method='mean')
            
            # Find best match from reference buffer
            best_match = None
            best_similarity = float('inf')
            
            for ref_vehicle in reference_vehicles_buffer:
                similarity = calculate_similarity(target_aggregated, ref_vehicle['features'], self.feature_method)
                
                if similarity < best_similarity and is_similarity_match(similarity, self.feature_method):
                    # Check if this reference vehicle is already assigned
                    already_assigned = any(
                        assignment.get('reference_id') == ref_vehicle['vehicle_id']
                        for assignment in self.target_assignments.values()
                    )
                    
                    if not already_assigned:
                        best_similarity = similarity
                        best_match = ref_vehicle
            
            # Assign result
            if best_match:
                self.target_assignments[track_id] = {
                    'vehicle_id': best_match['vehicle_id'],
                    'reference_id': best_match['vehicle_id'],
                    'tag': 'RE-ID',
                    'similarity': best_similarity,
                    'is_match': True
                }
                processing_stats['reidentified_vehicles'] += 1
                if args.debug:
                    print(f"🟢 RE-ID: Target Track {track_id} → {best_match['vehicle_id']} (similarity: {best_similarity:.3f})")
            else:
                new_vehicle_id = f"B{self.new_vehicle_counter}"
                self.target_assignments[track_id] = {
                    'vehicle_id': new_vehicle_id,
                    'reference_id': None,
                    'tag': 'NEW',
                    'similarity': 0.0,
                    'is_match': False
                }
                self.new_vehicle_counter += 1
                processing_stats['new_vehicles'] += 1
                if args.debug:
                    print(f"🔴 NEW: Target Track {track_id} → {new_vehicle_id}")
            
            processing_stats['total_target_vehicles'] = len(self.target_assignments)
    
    def get_reference_assignment(self, track_id):
        """Get reference vehicle assignment for display"""
        if track_id in self.ref_track_to_vehicle:
            vehicle_id = self.ref_track_to_vehicle[track_id]
            appearances = len(self.reference_tracks[track_id])
            return vehicle_id, appearances
        return None, 0
    
    def get_target_assignment(self, track_id):
        """Get target vehicle assignment for display"""
        if track_id in self.target_assignments:
            assignment = self.target_assignments[track_id]
            return assignment['vehicle_id'], assignment['tag'], assignment['similarity']
        return None, "PROCESSING", 0.0

def create_runs_directory():
    """Create runs_v3 directory for persistent database storage"""
    runs_dir = "runs_v3"
    os.makedirs(runs_dir, exist_ok=True)
    
    # Use fixed directory name - no incremental naming
    exp_name = "live_session"
    exp_dir = os.path.join(runs_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir, exp_name

def print_live_header():
    """Print a clean header for the live application"""
    print("=" * 80)
    print("📹 LIVE VEHICLE RE-IDENTIFICATION SYSTEM")
    print("    AI Based Smart Parking Management System")
    print("    Real-Time CCTV Processing with Live Display")
    print("    National Institute of Electronics (NIE)")
    print("    Developed by: Mohammad Saad Iqbal & Rafay Abrar")
    print("=" * 80)
    print(f"📅 Live Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_live_config(args, exp_name, exp_dir, reference_source, target_source):
    """Print live configuration details"""
    print("📋 LIVE CONFIGURATION")
    print("-" * 40)
    print(f"🎯 Session: {exp_name}")
    print(f"📁 Output Directory: {exp_dir}")
    
    # Show smart defaults configuration
    print("\n🔧 SMART DEFAULTS CONFIGURED:")
    print(f"   • Default Reference Video: {DEFAULT_REF_VID_A or 'None'}")
    print(f"   • Default Target Video: {DEFAULT_TGT_VID_B or 'None'}")
    print(f"   • Default Camera A: {DEFAULT_CAMERA_IP_A}")
    print(f"   • Default Camera B: {DEFAULT_CAMERA_IP_B}")
    
    # Show active sources
    ref_type = "📹 IP Camera" if reference_source.startswith('rtsp://') else "🎥 Video File"
    target_type = "📹 IP Camera" if target_source.startswith('rtsp://') else "🎥 Video File"
    
    print(f"\n📹 ACTIVE SOURCES:")
    print(f"   • Camera A (Reference): {ref_type}")
    print(f"     Source: {reference_source}")
    print(f"   • Camera B (Target): {target_type}")
    print(f"     Source: {target_source}")
    
    print(f"\n⚙️ PROCESSING SETTINGS:")
    print(f"   • Feature Method: {args.feature.upper()}")
    print(f"   • Display Scale: {args.display_scale}")
    print(f"   • FPS Limit: {args.fps_limit}")
    print(f"   • Frame Skip: Every {args.frame_skip} frames")
    print(f"   • Camera A Delay: {args.camera_a_delay}s")
    print(f"   • Camera B Delay: {args.camera_b_delay}s")
    print(f"⏭️  Frame Skip: {args.frame_skip} (process every {args.frame_skip} frames)")
    print(f"🔄 Buffer Size: {args.buffer_size}")
    
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
    
    print(f"💾 Save Video: {'Yes' if args.save_video else 'No (Live Mode)'}")
    if args.debug:
        print(f"🐛 Debug Mode: Enabled")
    print()

def process_reference_camera(video_path, tracker):
    """Process reference camera feed in separate thread with speed optimizations"""
    print("📹 Starting Reference Camera (Camera A) processing...")
    
    # Apply start delay if configured
    if args.camera_a_delay > 0:
        print(f"⏳ Camera A delay: Waiting {args.camera_a_delay} seconds before starting...")
        time.sleep(args.camera_a_delay)
        print("✅ Camera A delay completed, starting processing...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open reference video!")
        return
    
    # Speed optimizations for camera feeds
    if video_path.startswith('rtsp://'):
        print("🚀 Optimizing for RTSP camera feed...")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        cap.set(cv2.CAP_PROP_FPS, 15)        # Limit FPS at source
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    class_list = [2]  # Only cars
    
    # Frame skipping counter
    frame_skip_counter = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # For video files, stop when video ends (no looping)
            if not video_path.startswith('rtsp://'):
                print("📹 Reference video completed - stopping processing")
                break
            else:
                print("⚠️ Camera A connection lost, attempting to reconnect...")
                time.sleep(2)
                cap = cv2.VideoCapture(video_path)
                continue
        
        frame_count += 1
        frame_skip_counter += 1
        
        # Skip frames for speed optimization
        if frame_skip_counter % args.frame_skip != 0:
            continue
            
        fps_counter += 1
        
        # Limit FPS
        time.sleep(1.0 / args.fps_limit)
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if height > 720:  # Resize large frames for speed
            scale_factor = 720.0 / height
            new_width = int(width * scale_factor)
            new_height = 720
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Run detection and tracking with error handling
        try:
            # Use predict first, then track if needed (more stable)
            results = tracker.yolo_model.predict(frame, show=False, verbose=False, 
                                                conf=yolo_confidence, classes=class_list)
            
            # Try tracking if predict works
            if results and len(results) > 0:
                try:
                    results = tracker.yolo_model.track(frame, show=False, verbose=False, 
                                                     conf=yolo_confidence, classes=class_list, persist=True)
                except Exception as track_error:
                    if args.debug:
                        print(f"⚠️ Tracking failed for reference, using detection: {track_error}")
                    # Continue with detection results if tracking fails
                except:
                    # Fall back to prediction without tracking
                    if args.debug:
                        print("⚠️ Tracking failed, using detection only")
                    
        except Exception as e:
            if args.debug:
                print(f"⚠️ Reference camera YOLO error: {e}")
            # Skip this frame and continue
            continue
        
        # Process detections
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu()
            
            # Check if we have tracking IDs
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                # Generate temporary IDs if tracking failed
                track_ids = list(range(len(boxes)))
                if args.debug:
                    print("⚠️ No tracking IDs, using temporary IDs")
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                crop_img = frame[y1:y2, x1:x2]
                track_id = track_ids[i] if i < len(track_ids) else i
                
                # Extract features
                features, confidence = extract_features_with_confidence(crop_img, args.feature, tracker.feature_models)
                
                if features is not None:
                    tracker.update_reference_vehicle(track_id, frame_count, [x1, y1, x2, y2], features, confidence)
        
        # Add frame to queue for display
        if not reference_frame_queue.full():
            # Annotate frame
            annotated_frame = frame.copy()
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu()
                
                # Check if we have tracking IDs for display
                if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(boxes)))
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = track_ids[i] if i < len(track_ids) else i
                    
                    vehicle_id, appearances = tracker.get_reference_assignment(track_id)
                    if vehicle_id:
                        draw_live_reference_annotations(annotated_frame, x1, y1, x2, y2, 
                                                      vehicle_id, appearances, args.feature.upper())
            
            reference_frame_queue.put(annotated_frame)
        
        # Calculate FPS
        if time.time() - fps_start_time >= 1.0:
            processing_stats['reference_fps'] = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
    
    cap.release()

def process_target_camera(video_path, tracker):
    """Process target camera feed in separate thread with speed optimizations"""
    print("📹 Starting Target Camera (Camera B) processing...")
    
    # Apply start delay if configured
    if args.camera_b_delay > 0:
        print(f"⏳ Camera B delay: Waiting {args.camera_b_delay} seconds before starting...")
        time.sleep(args.camera_b_delay)
        print("✅ Camera B delay completed, starting processing...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open target video!")
        return
    
    # Speed optimizations for camera feeds
    if video_path.startswith('rtsp://'):
        print("🚀 Optimizing for RTSP camera feed...")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        cap.set(cv2.CAP_PROP_FPS, 15)        # Limit FPS at source
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    class_list = [2]  # Only cars
    
    # Frame skipping counter
    frame_skip_counter = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # For video files, stop when video ends (no looping)
            if not video_path.startswith('rtsp://'):
                print("📹 Target video completed - stopping processing")
                break
            else:
                print("⚠️ Camera B connection lost, attempting to reconnect...")
                time.sleep(2)
                cap = cv2.VideoCapture(video_path)
                continue
        
        frame_count += 1
        frame_skip_counter += 1
        
        # Skip frames for speed optimization
        if frame_skip_counter % args.frame_skip != 0:
            continue
            
        fps_counter += 1
        
        # Limit FPS
        time.sleep(1.0 / args.fps_limit)
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if height > 720:  # Resize large frames for speed
            scale_factor = 720.0 / height
            new_width = int(width * scale_factor)
            new_height = 720
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Run detection and tracking with error handling
        try:
            # Use predict first, then track if needed (more stable)
            results = tracker.yolo_model.predict(frame, show=False, verbose=False, 
                                                conf=yolo_confidence, classes=class_list)
            
            # Try tracking if predict works
            if results and len(results) > 0:
                try:
                    results = tracker.yolo_model.track(frame, show=False, verbose=False, 
                                                     conf=yolo_confidence, classes=class_list, persist=True)
                except Exception as track_error:
                    if args.debug:
                        print(f"⚠️ Tracking failed for target, using detection: {track_error}")
                    # Continue with detection results if tracking fails
                    
        except Exception as e:
            if args.debug:
                print(f"⚠️ Target camera YOLO error: {e}")
            # Skip this frame and continue
            continue
        
        # Process detections
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu()
            
            # Check if we have tracking IDs
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                # Generate temporary IDs if tracking failed
                track_ids = list(range(len(boxes)))
                if args.debug:
                    print("⚠️ No tracking IDs, using temporary IDs")
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                crop_img = frame[y1:y2, x1:x2]
                track_id = track_ids[i] if i < len(track_ids) else i
                
                # Extract features
                features, confidence = extract_features_with_confidence(crop_img, args.feature, tracker.feature_models)
                
                if features is not None:
                    tracker.process_target_vehicle(track_id, frame_count, [x1, y1, x2, y2], features, confidence)
        
        # Add frame to queue for display
        if not target_frame_queue.full():
            # Annotate frame
            annotated_frame = frame.copy()
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu()
                
                # Check if we have tracking IDs for display
                if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(boxes)))
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = track_ids[i] if i < len(track_ids) else i
                    
                    vehicle_id, tag, similarity = tracker.get_target_assignment(track_id)
                    if vehicle_id:
                        is_match = (tag == 'RE-ID')
                        draw_live_target_annotations(annotated_frame, x1, y1, x2, y2, 
                                                   vehicle_id, tag, similarity, args.feature.upper(), is_match)
            
            target_frame_queue.put(annotated_frame)
        
        # Calculate FPS
        if time.time() - fps_start_time >= 1.0:
            processing_stats['target_fps'] = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
    
    cap.release()

def display_live_feeds():
    """Display live feeds from both cameras with statistics"""
    print("📺 Starting live display...")
    
    # Create windows
    cv2.namedWindow('Camera A - Reference Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Camera B - Target Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Live Statistics', cv2.WINDOW_NORMAL)
    
    # Position windows
    cv2.moveWindow('Camera A - Reference Feed', 100, 100)
    cv2.moveWindow('Camera B - Target Feed', 700, 100)
    cv2.moveWindow('Live Statistics', 100, 500)
    
    stats_display_counter = 0
    
    while True:
        display_success = False
        
        # Display reference feed
        if not reference_frame_queue.empty():
            ref_frame = reference_frame_queue.get()
            
            # Resize for display
            height, width = ref_frame.shape[:2]
            new_width = int(width * args.display_scale)
            new_height = int(height * args.display_scale)
            ref_frame_resized = cv2.resize(ref_frame, (new_width, new_height))
            
            # Add title overlay
            cv2.putText(ref_frame_resized, "CAMERA A - REFERENCE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Camera A - Reference Feed', ref_frame_resized)
            display_success = True
        
        # Display target feed
        if not target_frame_queue.empty():
            target_frame = target_frame_queue.get()
            
            # Resize for display
            height, width = target_frame.shape[:2]
            new_width = int(width * args.display_scale)
            new_height = int(height * args.display_scale)
            target_frame_resized = cv2.resize(target_frame, (new_width, new_height))
            
            # Add title overlay
            cv2.putText(target_frame_resized, "CAMERA B - TARGET", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Camera B - Target Feed', target_frame_resized)
            display_success = True
        
        # Display statistics every 30 frames
        stats_display_counter += 1
        if stats_display_counter % 30 == 0:
            stats_frame = create_statistics_display()
            cv2.imshow('Live Statistics', stats_frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        
        # If no frames to display, small delay to prevent busy waiting
        if not display_success:
            time.sleep(0.01)
    
    cv2.destroyAllWindows()

def create_statistics_display():
    """Create an optimized statistics display frame"""
    # Create a black frame for statistics
    stats_frame = np.zeros((450, 650, 3), dtype=np.uint8)
    
    # Title
    cv2.putText(stats_frame, "LIVE RE-ID STATISTICS - OPTIMIZED", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Current time
    current_time = datetime.now().strftime('%H:%M:%S')
    cv2.putText(stats_frame, f"Time: {current_time}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Source information
    y_offset = 85
    
    # Show source types
    ref_type = "Camera" if REFERENCE_SOURCE.startswith('rtsp://') else "Video"
    target_type = "Camera" if TARGET_SOURCE.startswith('rtsp://') else "Video"
    
    cv2.putText(stats_frame, f"Reference: {ref_type} | Target: {target_type}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_offset += 20
    
    # Performance stats
    y_offset += 10
    performance_stats = [
        f"Camera A FPS: {processing_stats['reference_fps']}",
        f"Camera B FPS: {processing_stats['target_fps']}",
        f"Processing FPS: {min(processing_stats['reference_fps'], processing_stats['target_fps'])}",
        f"Frame Skip: 1/{args.frame_skip} (Speed Boost)",
    ]
    
    for stat in performance_stats:
        cv2.putText(stats_frame, stat, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 25
    
    # Separation line
    cv2.line(stats_frame, (10, y_offset + 5), (640, y_offset + 5), (128, 128, 128), 1)
    y_offset += 20
    
    # Vehicle stats
    vehicle_stats = [
        f"Reference Vehicles: {processing_stats['total_ref_vehicles']}",
        f"Target Vehicles: {processing_stats['total_target_vehicles']}",
        f"Re-identified: {processing_stats['reidentified_vehicles']}",
        f"New Vehicles: {processing_stats['new_vehicles']}",
    ]
    
    for stat in vehicle_stats:
        cv2.putText(stats_frame, stat, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    # Separation line
    cv2.line(stats_frame, (10, y_offset + 5), (640, y_offset + 5), (128, 128, 128), 1)
    y_offset += 20
    
    # Buffer status
    buffer_stats = [
        f"Reference Buffer: {len(reference_vehicles_buffer)}/{args.buffer_size}",
        f"Reference Queue: {reference_frame_queue.qsize()}/5",
        f"Target Queue: {target_frame_queue.qsize()}/5",
        f"Display Scale: {args.display_scale} (Performance Mode)",
    ]
    
    for stat in buffer_stats:
        cv2.putText(stats_frame, stat, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 25
    
    # Instructions
    y_offset += 15
    cv2.putText(stats_frame, "Press 'q' or ESC to quit", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Speed optimization indicators
    y_offset += 25
    cv2.putText(stats_frame, "SPEED OPTIMIZATIONS ACTIVE", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    y_offset += 20
    cv2.putText(stats_frame, f"• Frame skip: {args.frame_skip}x faster", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    y_offset += 15
    cv2.putText(stats_frame, f"• Display: {args.display_scale}x scale", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    y_offset += 15
    cv2.putText(stats_frame, f"• FPS limit: {args.fps_limit}", (20, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    
    return stats_frame
    
    for stat in vehicle_stats:
        cv2.putText(stats_frame, stat, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    # Separation line
    cv2.line(stats_frame, (10, y_offset + 5), (590, y_offset + 5), (128, 128, 128), 1)
    y_offset += 20
    
    # Buffer status
    buffer_stats = [
        f"Reference Buffer: {len(reference_vehicles_buffer)}/{args.buffer_size}",
        f"Reference Queue: {reference_frame_queue.qsize()}/5",
        f"Target Queue: {target_frame_queue.qsize()}/5",
    ]
    
    for stat in buffer_stats:
        cv2.putText(stats_frame, stat, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 25
    
    # Instructions
    y_offset += 20
    cv2.putText(stats_frame, "Press 'q' or ESC to quit", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return stats_frame

def draw_live_reference_annotations(frame, x1, y1, x2, y2, vehicle_id, appearances, model_name):
    """Draw annotations on the live reference feed"""
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
    
    # Upper Right Corner: Appearances
    app_text = f"App: {appearances}"
    (app_w, app_h), _ = cv2.getTextSize(app_text, font, font_scale, thickness)
    cv2.rectangle(frame, (x2-app_w-5, y1-app_h-5), (x2, y1), bg_color, -1)
    cv2.putText(frame, app_text, (x2-app_w-2, y1-5), font, font_scale, text_color, thickness)
    
    # Bottom Left Corner: Model Name
    model_text = f"REF-{model_name}"
    (model_w, model_h), _ = cv2.getTextSize(model_text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y2), (x1+model_w+5, y2+model_h+5), bg_color, -1)
    cv2.putText(frame, model_text, (x1+2, y2+model_h), font, font_scale, text_color, thickness)

def draw_live_target_annotations(frame, x1, y1, x2, y2, vehicle_id, tag, similarity, model_name, is_match):
    """Draw annotations on the live target feed"""
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
    id_text = vehicle_id if vehicle_id else "PROCESSING"
    (id_w, id_h), _ = cv2.getTextSize(id_text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1-id_h-5), (x1+id_w+5, y1), bg_color, -1)
    cv2.putText(frame, id_text, (x1+2, y1-5), font, font_scale, text_color, thickness)
    
    # Upper Right Corner: Tag and Similarity
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
    elif feature_method in ["resnet", "mobilenet"]:
        return similarity_score < cosine_threshold
    elif feature_method == "composite":
        return similarity_score < composite_threshold
    elif feature_method == "sift":
        return similarity_score < sift_threshold
    
    return False

# ===============================================================================
# SIMILARITY THRESHOLDS - Adjust these values to control matching sensitivity
# Lower values = more strict matching (fewer matches but higher confidence)
# Higher values = more lenient matching (more matches but lower confidence)
# ===============================================================================

default_hog_threshold = 0.5          # HOG L2 distance threshold (lower = stricter)
default_histogram_threshold = 0.85   # Color histogram correlation threshold (higher = stricter)
default_resnet_threshold = 0.375     # ResNet cosine distance threshold (lower = stricter)
default_mobilenet_threshold = 0.3    # MobileNetV2 cosine distance threshold (lower = stricter)
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
    """Main function for live processing with hybrid camera/video support"""
    # Create experiment directory
    exp_dir, exp_name = create_runs_directory()
    
    # Print header and configuration
    print_live_header()
    print_live_config(args, exp_name, exp_dir, REFERENCE_SOURCE, TARGET_SOURCE)
    
    # Initialize live tracker
    print("🔧 Initializing live vehicle tracker...")
    tracker = LiveVehicleTracker(args.feature, exp_dir)
    
    try:
        print("🚀 Starting live processing threads...")
        
        # Show delay information if configured
        if args.camera_a_delay > 0 or args.camera_b_delay > 0:
            print("⏱️  CAMERA START DELAYS CONFIGURED:")
            if args.camera_a_delay > 0:
                print(f"   • Camera A will start after {args.camera_a_delay} seconds")
            if args.camera_b_delay > 0:
                print(f"   • Camera B will start after {args.camera_b_delay} seconds")
            print("   • This helps with accurate result analysis")
            print()
        
        # Start processing threads with actual sources
        ref_thread = threading.Thread(target=process_reference_camera, args=(REFERENCE_SOURCE, tracker))
        target_thread = threading.Thread(target=process_target_camera, args=(TARGET_SOURCE, tracker))
        
        ref_thread.daemon = True
        target_thread.daemon = True
        
        ref_thread.start()
        target_thread.start()
        
        # Give threads time to start
        time.sleep(3)  # Slightly longer for camera connections
        
        print("📺 Starting optimized live display...")
        print("=" * 80)
        print("📹 LIVE FEEDS ACTIVE - Press 'q' or ESC to quit")
        print("🚀 Performance optimizations enabled:")
        print(f"   • Frame skipping: Every {args.frame_skip} frames")
        print(f"   • Display scale: {args.display_scale}")
        print(f"   • FPS limit: {args.fps_limit}")
        print("=" * 80)
        
        # Start display (this will block until user quits)
        display_live_feeds()
        
        print("\n⏹️  Live processing stopped by user")
        
    except KeyboardInterrupt:
        print("\n⚠️ Live processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in live processing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        print("🧹 Cleaning up...")
        
        # Clean up database after session completion
        tracker.cleanup_database()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n📊 FINAL STATISTICS:")
        print(f"   • Reference vehicles processed: {processing_stats['total_ref_vehicles']}")
        print(f"   • Target vehicles processed: {processing_stats['total_target_vehicles']}")
        print(f"   • Successfully re-identified: {processing_stats['reidentified_vehicles']}")
        print(f"   • New vehicles found: {processing_stats['new_vehicles']}")
        print(f"   • Database cleaned after session")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Application interrupted by user")
    except Exception as e:
        print(f"\n❌ Application error: {e}")

# =====================================================================================
# USAGE EXAMPLES - Smart Default System with Camera Delays
# =====================================================================================
# 1. Use configured defaults (videos → cameras fallback):
#    python main_live_reid.py
# 
# 2. Override with specific videos:
#    python main_live_reid.py --reference-video v1.mp4 --target-video v2.mp4
# 
# 3. Mixed sources (video + camera):
#    python main_live_reid.py --reference-video v1.mp4  # target uses default
# 
# 4. Force IP camera mode:
#    python main_live_reid.py --use-cameras
# 
# 5. Speed optimizations:
#    python main_live_reid.py --frame-skip 3 --fps-limit 20 --display-scale 0.6
#
# 6. Camera start delays (for accurate result analysis):
#    python main_live_reid.py --camera-a-delay 0 --camera-b-delay 10
#    python main_live_reid.py --reference-video v1.mp4 --target-video v2.mp4 --camera-b-delay 15
#
# 7. Complete example with all options:
#    python main_live_reid.py --reference-video v1.mp4 --camera-b-delay 5 --frame-skip 2 --fps-limit 20
# =====================================================================================
    except KeyboardInterrupt:
        print("\n⚠️ Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
