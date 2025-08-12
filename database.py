"""
@brief: Database Module for Vehicle Re-identification Feature Storage
@project: AI Based Smart Parking Management System - Database Component
@authors: Mohammad Saad Iqbal, Rafay Abrar
@organization: National Institute of Electronics (NIE)
@description: SQLite database management for storing vehicle features, tracks, and metadata
             supporting multi-video vehicle re-identification with feature aggregation.
"""

import sqlite3
import json
import numpy as np
import os
from datetime import datetime
import pickle

class VehicleDatabase:
    def __init__(self, db_path="vehicle_database.db"):
        """Initialize database connection and create tables if they don't exist"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        """Create necessary tables for vehicle re-identification"""
        try:
            cursor = self.conn.cursor()
            # Table for storing video information
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT UNIQUE NOT NULL,
                    video_type TEXT NOT NULL,  -- 'reference' or 'target'
                    feature_method TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_frames INTEGER,
                    total_vehicles INTEGER
                )
            ''')
            # Table for storing vehicle tracks and metadata
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    vehicle_id TEXT NOT NULL,  -- A0, A1, B0, B1, etc.
                    track_id INTEGER,
                    first_frame INTEGER,
                    last_frame INTEGER,
                    total_appearances INTEGER,
                    bbox_data TEXT,  -- JSON string of all bounding boxes
                    FOREIGN KEY (video_id) REFERENCES videos (id),
                    UNIQUE(video_id, vehicle_id)
                )
            ''')
            # Table for storing extracted features for each vehicle appearance
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicle_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_table_id INTEGER,
                    frame_number INTEGER,
                    features BLOB,  -- Pickled numpy array
                    confidence REAL,
                    feature_type TEXT,
                    FOREIGN KEY (vehicle_table_id) REFERENCES vehicles (id)
                )
            ''')
            # Table for storing aggregated features per vehicle
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS aggregated_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_table_id INTEGER,
                    aggregated_features BLOB,  -- Pickled numpy array
                    feature_type TEXT,
                    num_frames INTEGER,
                    avg_confidence REAL,
                    FOREIGN KEY (vehicle_table_id) REFERENCES vehicles (id),
                    UNIQUE(vehicle_table_id, feature_type)
                )
            ''')
            # Table for storing re-identification results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reid_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_vehicle_id INTEGER,
                    reference_vehicle_id INTEGER,
                    similarity_score REAL,
                    is_match BOOLEAN,
                    feature_method TEXT,
                    FOREIGN KEY (target_vehicle_id) REFERENCES vehicles (id),
                    FOREIGN KEY (reference_vehicle_id) REFERENCES vehicles (id)
                )
            ''')
            self.conn.commit()
        except Exception as e:
            print(f"[DB ERROR] create_tables: {e}")
        finally:
            cursor.close()
    
    def add_video(self, video_path, video_type, feature_method, total_frames=0, total_vehicles=0):
        """Add a new video to the database"""
        try:
            cursor = self.conn.cursor()
            timestamp = datetime.now().isoformat()
            cursor.execute('''
                INSERT OR REPLACE INTO videos 
                (video_path, video_type, feature_method, timestamp, total_frames, total_vehicles)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (video_path, video_type, feature_method, timestamp, total_frames, total_vehicles))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"[DB ERROR] add_video: {e}")
            return None
        finally:
            cursor.close()
    
    def add_vehicle(self, video_id, vehicle_id, track_id, first_frame, last_frame, total_appearances, bbox_data):
        """Add a new vehicle track to the database"""
        try:
            cursor = self.conn.cursor()
            bbox_json = json.dumps(bbox_data)
            cursor.execute('''
                INSERT OR REPLACE INTO vehicles 
                (video_id, vehicle_id, track_id, first_frame, last_frame, total_appearances, bbox_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (video_id, vehicle_id, track_id, first_frame, last_frame, total_appearances, bbox_json))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"[DB ERROR] add_vehicle: {e}")
            return None
        finally:
            cursor.close()
    
    def add_vehicle_feature(self, vehicle_table_id, frame_number, features, confidence, feature_type):
        """Add features for a specific vehicle appearance"""
        try:
            cursor = self.conn.cursor()
            features_blob = pickle.dumps(features)
            cursor.execute('''
                INSERT INTO vehicle_features 
                (vehicle_table_id, frame_number, features, confidence, feature_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (vehicle_table_id, frame_number, features_blob, confidence, feature_type))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"[DB ERROR] add_vehicle_feature: {e}")
            return None
        finally:
            cursor.close()
    
    def add_aggregated_features(self, vehicle_table_id, aggregated_features, feature_type, num_frames, avg_confidence):
        """Add aggregated features for a vehicle"""
        try:
            cursor = self.conn.cursor()
            features_blob = pickle.dumps(aggregated_features)
            cursor.execute('''
                INSERT OR REPLACE INTO aggregated_features 
                (vehicle_table_id, aggregated_features, feature_type, num_frames, avg_confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (vehicle_table_id, features_blob, feature_type, num_frames, avg_confidence))
            self.conn.commit()
        except Exception as e:
            print(f"[DB ERROR] add_aggregated_features: {e}")
        finally:
            cursor.close()
    
    def get_reference_vehicles(self, feature_method):
        """Get all reference vehicles with their aggregated features"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT v.vehicle_id, v.track_id, af.aggregated_features, af.avg_confidence, v.id
            FROM vehicles v
            JOIN videos vid ON v.video_id = vid.id
            JOIN aggregated_features af ON v.id = af.vehicle_table_id
            WHERE vid.video_type = 'reference' AND vid.feature_method = ? AND af.feature_type = ?
        ''', (feature_method, feature_method))
        
        results = []
        for row in cursor.fetchall():
            vehicle_id, track_id, features_blob, avg_confidence, table_id = row
            features = pickle.loads(features_blob)
            results.append({
                'vehicle_id': vehicle_id,
                'track_id': track_id,
                'features': features,
                'confidence': avg_confidence,
                'table_id': table_id
            })
        
        return results
    
    def get_vehicle_by_id(self, video_id, vehicle_id):
        """Get vehicle information by video_id and vehicle_id"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, track_id, first_frame, last_frame, total_appearances, bbox_data
            FROM vehicles
            WHERE video_id = ? AND vehicle_id = ?
        ''', (video_id, vehicle_id))
        
        result = cursor.fetchone()
        if result:
            table_id, track_id, first_frame, last_frame, total_appearances, bbox_json = result
            bbox_data = json.loads(bbox_json)
            return {
                'table_id': table_id,
                'track_id': track_id,
                'first_frame': first_frame,
                'last_frame': last_frame,
                'total_appearances': total_appearances,
                'bbox_data': bbox_data
            }
        return None
    
    def add_reid_result(self, target_vehicle_id, reference_vehicle_id, similarity_score, is_match, feature_method):
        """Add re-identification result"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO reid_results 
                (target_vehicle_id, reference_vehicle_id, similarity_score, is_match, feature_method)
                VALUES (?, ?, ?, ?, ?)
            ''', (target_vehicle_id, reference_vehicle_id, similarity_score, is_match, feature_method))
            self.conn.commit()
        except Exception as e:
            print(f"[DB ERROR] add_reid_result: {e}")
        finally:
            cursor.close()
    
    def get_best_match(self, target_vehicle_table_id, feature_method):
        """Get the best match for a target vehicle"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT rr.reference_vehicle_id, rr.similarity_score, rr.is_match, v.vehicle_id
            FROM reid_results rr
            JOIN vehicles v ON rr.reference_vehicle_id = v.id
            WHERE rr.target_vehicle_id = ? AND rr.feature_method = ?
            ORDER BY rr.similarity_score ASC
            LIMIT 1
        ''', (target_vehicle_table_id, feature_method))
        
        result = cursor.fetchone()
        if result:
            ref_table_id, similarity_score, is_match, ref_vehicle_id = result
            return {
                'ref_vehicle_id': ref_vehicle_id,
                'similarity_score': similarity_score,
                'is_match': bool(is_match)
            }
        return None
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def clear_database(self):
        """Clear all data from database (for testing)"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM reid_results')
        cursor.execute('DELETE FROM aggregated_features')
        cursor.execute('DELETE FROM vehicle_features')
        cursor.execute('DELETE FROM vehicles')
        cursor.execute('DELETE FROM videos')
        self.conn.commit()

def aggregate_features(features_list, method='mean'):
    """Aggregate multiple feature vectors"""
    if not features_list:
        return None
    
    if method == 'mean':
        return np.mean(features_list, axis=0)
    elif method == 'median':
        return np.median(features_list, axis=0)
    elif method == 'max':
        return np.max(features_list, axis=0)
    else:
        return np.mean(features_list, axis=0)
