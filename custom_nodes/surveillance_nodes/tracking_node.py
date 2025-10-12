"""
Object Tracking Node using ByteTrack
Tracks detected objects across frames to maintain identity
"""

import os
import numpy as np
import uuid
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ByteTrack imports
try:
    from bytetrack.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("Warning: ByteTrack not available. Using simple IoU tracker as fallback")

class SimpleTracker:
    """Simple IoU-based tracker as fallback when ByteTrack not available"""
    
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 0
        self.max_age = 30  # frames
        
    def update(self, detections):
        """Update tracks with new detections"""
        updated_tracks = []
        
        if not detections:
            # Age out tracks
            for track_id, track in list(self.tracks.items()):
                track['age'] += 1
                if track['age'] > self.max_age:
                    del self.tracks[track_id]
            return []
        
        # Convert detections to numpy array
        det_boxes = []
        for det in detections:
            bbox = det['attributes']['pixel_bbox']
            det_boxes.append([
                bbox['x1'], bbox['y1'], 
                bbox['x2'], bbox['y2'],
                det['confidence']
            ])
        det_boxes = np.array(det_boxes) if det_boxes else np.empty((0, 5))
        
        # Simple matching based on IoU
        matched = []
        unmatched_dets = list(range(len(detections)))
        
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det_idx = -1
            
            for det_idx in unmatched_dets:
                iou = self._compute_iou(track['bbox'], det_boxes[det_idx][:4])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                matched.append((track_id, best_det_idx))
                unmatched_dets.remove(best_det_idx)
        
        # Update matched tracks
        for track_id, det_idx in matched:
            self.tracks[track_id]['bbox'] = det_boxes[det_idx][:4]
            self.tracks[track_id]['age'] = 0
            self.tracks[track_id]['detection'] = detections[det_idx]
            updated_tracks.append({
                'track_id': track_id,
                'bbox': det_boxes[det_idx][:4],
                'detection': detections[det_idx]
            })
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                'bbox': det_boxes[det_idx][:4],
                'age': 0,
                'detection': detections[det_idx]
            }
            updated_tracks.append({
                'track_id': track_id,
                'bbox': det_boxes[det_idx][:4],
                'detection': detections[det_idx]
            })
        
        return updated_tracks
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou

class TrackingNode:
    """Track objects across frames to maintain identity"""
    
    def __init__(self):
        self.tracker = None
        self.track_history = defaultdict(list)
        self.active_tracks = {}
        self.storage_root = Path("./surveillance_storage")
        self.frames_dir = self.storage_root / "frames"
            
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detections": ("DETECTIONS",),
                "detection_metadata": ("DETECTION_METADATA",),
                "track_thresh": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Tracking confidence threshold"
                }),
                "track_buffer": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "tooltip": "Frames to keep track alive without detection"
                }),
                "min_track_length": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Minimum detections to consider valid track"
                }),
            },
            "optional": {
                "class_filter": ("STRING", {
                    "default": "person",
                    "tooltip": "Classes to track (comma-separated) or 'all'"
                }),
            }
        }
    
    RETURN_TYPES = ("TRACKS", "TRACK_METADATA")
    RETURN_NAMES = ("tracks", "track_metadata")
    FUNCTION = "track_objects"
    CATEGORY = "Surveillance/Tracking"
    
    def _init_tracker(self, track_thresh: float, track_buffer: int):
        """Initialize tracker based on availability"""
        if BYTETRACK_AVAILABLE:
            print("Using ByteTrack tracker")
            tracker_config = {
                'track_thresh': track_thresh,
                'track_buffer': track_buffer,
                'match_thresh': 0.8,
                'mot20': False
            }
            return BYTETracker(**tracker_config)
        else:
            print("Using simple IoU tracker (install byte-track for better performance)")
            return SimpleTracker(iou_threshold=track_thresh)
    
    def track_objects(self, detections: List[Dict], detection_metadata: Dict,
                     track_thresh: float, track_buffer: int, min_track_length: int,
                     class_filter: str = "person") -> Tuple[List, Dict]:
        """
        Track objects across frames
        
        Returns:
            tracks: List of track objects with trajectories
            track_metadata: Tracking statistics and analysis
        """
        
        # Initialize tracker
        if self.tracker is None:
            self.tracker = self._init_tracker(track_thresh, track_buffer)
        
        # Parse class filter
        if class_filter.lower() == "all":
            target_classes = None
        else:
            target_classes = [c.strip() for c in class_filter.split(',')]
        
        # Group detections by frame
        frame_detections = defaultdict(list)
        for det in detections:
            if target_classes and det['class_name'] not in target_classes:
                continue
            frame_detections[det['frame_index']].append(det)
        
        # Sort frame indices
        sorted_frames = sorted(frame_detections.keys())
        
        print(f"Tracking objects across {len(sorted_frames)} frames...")
        
        # Track objects frame by frame
        all_tracks = defaultdict(lambda: {
            'track_id': None,
            'detections': [],
            'trajectory': [],
            'frame_indices': [],
            'start_frame': None,
            'end_frame': None,
            'class_name': None
        })
        
        for frame_idx in sorted_frames:
            frame_dets = frame_detections[frame_idx]
            
            # Update tracker
            if BYTETRACK_AVAILABLE and frame_dets:
                # Convert to ByteTrack format
                det_array = []
                for det in frame_dets:
                    bbox = det['attributes']['pixel_bbox']
                    det_array.append([
                        bbox['x1'], bbox['y1'],
                        bbox['x2'], bbox['y2'],
                        det['confidence']
                    ])
                det_array = np.array(det_array)
                
                # Update tracker
                online_tracks = self.tracker.update(det_array, [1080, 1920], (1080, 1920))
                
                # Process tracks
                for track in online_tracks:
                    track_id = track.track_id
                    if track_id not in all_tracks:
                        all_tracks[track_id]['track_id'] = track_id
                    
                    # Find matching detection
                    for i, det in enumerate(frame_dets):
                        det_bbox = det['attributes']['pixel_bbox']
                        track_bbox = track.tlbr  # top-left bottom-right
                        
                        # Check if bboxes match (with small tolerance)
                        if (abs(det_bbox['x1'] - track_bbox[0]) < 5 and
                            abs(det_bbox['y1'] - track_bbox[1]) < 5):
                            
                            # Add detection to track
                            all_tracks[track_id]['detections'].append(det)
                            all_tracks[track_id]['frame_indices'].append(frame_idx)
                            all_tracks[track_id]['trajectory'].append({
                                'x': float((track_bbox[0] + track_bbox[2]) / 2),
                                'y': float((track_bbox[1] + track_bbox[3]) / 2),
                                'timestamp': det['timestamp'],
                                'frame_index': frame_idx
                            })
                            
                            if all_tracks[track_id]['start_frame'] is None:
                                all_tracks[track_id]['start_frame'] = frame_idx
                                all_tracks[track_id]['class_name'] = det['class_name']
                            all_tracks[track_id]['end_frame'] = frame_idx
                            break
            
            else:
                # Use simple tracker
                tracked = self.tracker.update(frame_dets)
                
                for track_info in tracked:
                    track_id = track_info['track_id']
                    det = track_info['detection']
                    
                    if track_id not in all_tracks:
                        all_tracks[track_id]['track_id'] = track_id
                    
                    all_tracks[track_id]['detections'].append(det)
                    all_tracks[track_id]['frame_indices'].append(frame_idx)
                    
                    bbox = det['attributes']['pixel_bbox']
                    all_tracks[track_id]['trajectory'].append({
                        'x': float((bbox['x1'] + bbox['x2']) / 2),
                        'y': float((bbox['y1'] + bbox['y2']) / 2),
                        'timestamp': det['timestamp'],
                        'frame_index': frame_idx
                    })
                    
                    if all_tracks[track_id]['start_frame'] is None:
                        all_tracks[track_id]['start_frame'] = frame_idx
                        all_tracks[track_id]['class_name'] = det['class_name']
                    all_tracks[track_id]['end_frame'] = frame_idx
        
        # Filter tracks by minimum length
        valid_tracks = []
        for track_id, track_data in all_tracks.items():
            if len(track_data['detections']) >= min_track_length:
                # Calculate track duration
                if track_data['detections']:
                    start_time = track_data['detections'][0]['timestamp']
                    end_time = track_data['detections'][-1]['timestamp']
                    duration = end_time - start_time
                else:
                    start_time = end_time = duration = 0
                
                track_obj = {
                    'track_id': track_id,
                    'detections': track_data['detections'],
                    'trajectory': track_data['trajectory'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'start_frame': track_data['start_frame'],
                    'end_frame': track_data['end_frame'],
                    'frame_count': len(track_data['detections']),
                    'class_name': track_data['class_name'],
                    'attributes': {
                        'avg_confidence': np.mean([d['confidence'] for d in track_data['detections']]),
                        'frame_indices': track_data['frame_indices']
                    }
                }
                valid_tracks.append(track_obj)
        
        # Analyze tracks for suspicious behavior
        suspicious_tracks = self._analyze_track_behavior(valid_tracks, detection_metadata)
        
        # Create track metadata
        track_metadata = {
            'session_id': detection_metadata['session_id'],
            'timestamp': datetime.now().isoformat(),
            'total_tracks': len(valid_tracks),
            'tracking_settings': {
                'track_thresh': track_thresh,
                'track_buffer': track_buffer,
                'min_track_length': min_track_length,
                'class_filter': class_filter
            },
            'track_statistics': {
                'avg_track_length': np.mean([t['frame_count'] for t in valid_tracks]) if valid_tracks else 0,
                'max_track_length': max([t['frame_count'] for t in valid_tracks]) if valid_tracks else 0,
                'avg_duration': np.mean([t['duration'] for t in valid_tracks]) if valid_tracks else 0,
                'total_unique_objects': len(valid_tracks)
            },
            'class_distribution': self._get_class_distribution(valid_tracks),
            'suspicious_analysis': suspicious_tracks,
            'all_tracks': valid_tracks
        }
        
        # Save metadata with correct path
        session_id = detection_metadata.get('session_id', 'default')
        session_dir = self.frames_dir / str(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist
        metadata_path = session_dir / "tracking_metadata.json"

        with open(metadata_path, 'w') as f:
            # Save subset without full track data
            save_meta = {k: v for k, v in track_metadata.items() if k != 'all_tracks'}
            save_meta['track_summaries'] = [{
                'track_id': t['track_id'],
                'class_name': t['class_name'],
                'duration': t['duration'],
                'frame_count': t['frame_count']
            } for t in valid_tracks]
            json.dump(save_meta, f, indent=2)
        
        print(f"✓ Generated {len(valid_tracks)} tracks from {len(detections)} detections")
        
        return (valid_tracks, track_metadata)
    
    def _analyze_track_behavior(self, tracks: List[Dict], detection_metadata: Dict) -> Dict:
        """Analyze tracks for suspicious behavior patterns"""
        
        suspicious_patterns = {
            'loitering': [],  # Stays in area too long
            'erratic_movement': [],  # Unusual movement patterns
            'abandoned_object': [],  # Object left behind
            'converging_tracks': [],  # Multiple tracks converging
            'rapid_appearance_disappearance': []  # Quick in and out
        }
        
        for track in tracks:
            # Loitering detection (person staying > 30 seconds)
            if track['class_name'] == 'person' and track['duration'] > 30:
                suspicious_patterns['loitering'].append({
                    'track_id': track['track_id'],
                    'duration': track['duration'],
                    'severity': 'medium' if track['duration'] < 60 else 'high'
                })
            
            # Erratic movement detection
            if len(track['trajectory']) > 5:
                trajectory = track['trajectory']
                velocities = []
                for i in range(1, len(trajectory)):
                    dx = trajectory[i]['x'] - trajectory[i-1]['x']
                    dy = trajectory[i]['y'] - trajectory[i-1]['y']
                    dt = trajectory[i]['timestamp'] - trajectory[i-1]['timestamp']
                    if dt > 0:
                        velocity = np.sqrt(dx**2 + dy**2) / dt
                        velocities.append(velocity)
                
                if velocities:
                    velocity_variance = np.var(velocities)
                    if velocity_variance > 100:  # High variance indicates erratic movement
                        suspicious_patterns['erratic_movement'].append({
                            'track_id': track['track_id'],
                            'variance': float(velocity_variance),
                            'severity': 'low' if velocity_variance < 200 else 'medium'
                        })
            
            # Abandoned object detection (non-person objects with long duration)
            if track['class_name'] in ['backpack', 'suitcase', 'handbag'] and track['duration'] > 60:
                suspicious_patterns['abandoned_object'].append({
                    'track_id': track['track_id'],
                    'object_type': track['class_name'],
                    'duration': track['duration'],
                    'severity': 'high'
                })
            
            # Rapid appearance/disappearance (very short tracks)
            if track['duration'] < 2 and track['frame_count'] < 5:
                suspicious_patterns['rapid_appearance_disappearance'].append({
                    'track_id': track['track_id'],
                    'duration': track['duration'],
                    'severity': 'low'
                })
        
        return suspicious_patterns
    
    def _get_class_distribution(self, tracks: List[Dict]) -> Dict[str, int]:
        """Get distribution of tracked object classes"""
        distribution = {}
        for track in tracks:
            class_name = track['class_name']
            if class_name not in distribution:
                distribution[class_name] = 0
            distribution[class_name] += 1
        return distribution

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "TrackingNode": TrackingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrackingNode": "Object Tracking (ByteTrack)"
}