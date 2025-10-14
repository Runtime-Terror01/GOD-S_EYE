"""
Alert Node - Rule-based alert generation
Analyzes detections and tracks to generate security alerts
"""

from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import uuid

from .base_node import BaseNode
from .data_types import CLASS_BATCH, BBOX_BATCH, CONF_BATCH, TRACKS_BATCH, FRAME_META_BATCH, ALERTS


class AlertNode(BaseNode):
    """
    Generate security alerts based on detections and tracking data
    Implements rule-based alert logic
    Outputs ALERTS list
    Saves alert.json
    """
    
    # Default alert rules
    DEFAULT_RULES = {
        "loitering": {
            "enabled": True,
            "duration_threshold": 30.0,  # seconds
            "severity": "Medium"
        },
        "suspicious_object": {
            "enabled": True,
            "watchlist": ["knife", "gun", "rifle", "pistol", "weapon"],
            "conf_threshold": 0.5,
            "severity": "Critical"
        },
        "crowding": {
            "enabled": True,
            "person_threshold": 10,
            "severity": "Medium"
        },
        "restricted_zone": {
            "enabled": False,  # Requires zone configuration
            "severity": "High"
        }
    }
    
    # ComfyUI Node Configuration
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "classes": ("CLASS_BATCH",),
                "bboxes": ("BBOX_BATCH",),
                "confidences": ("CONF_BATCH",),
                "tracks": ("TRACKS_BATCH",),
                "frame_metadata": ("FRAME_META_BATCH",),
            }
        }
    
    RETURN_TYPES = ("ALERTS",)
    RETURN_NAMES = ("alerts",)
    FUNCTION = "generate_alerts"
    CATEGORY = "surveillance"
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.rules = self.DEFAULT_RULES.copy()
        if config and 'rules' in config:
            self.rules.update(config['rules'])
        
        self.session_id = None
        self.track_history = defaultdict(list)  # Track positions over time
    
    def generate_alerts(self, classes, bboxes, confidences, tracks, frame_metadata):
        """ComfyUI entry point"""
        return self.process(classes, bboxes, confidences, tracks, frame_metadata)
        
    def process(self, class_batch: CLASS_BATCH, bbox_batch: BBOX_BATCH, 
                conf_batch: CONF_BATCH, tracks_batch: TRACKS_BATCH, 
                frames_meta: FRAME_META_BATCH) -> Tuple[ALERTS]:
        """
        Generate alerts based on detections and tracks
        
        Args:
            class_batch: List[List[str]] - detected classes
            bbox_batch: List[List[List[int]]] - bounding boxes
            conf_batch: List[List[float]] - confidence scores
            tracks_batch: List[List[dict]] - tracking data
            frames_meta: Frame metadata
            
        Returns:
            alerts: List of alert dictionaries
        """
        self.session_id = frames_meta[0]['session_id'] if frames_meta else "unknown"
        
        alerts = []
        
        print(f"Analyzing {len(frames_meta)} frames for alerts...")
        
        # Build track history
        self._build_track_history(tracks_batch, frames_meta)
        
        # Check rules for each frame
        for idx, (classes, bboxes, confs, tracks, meta) in enumerate(
            zip(class_batch, bbox_batch, conf_batch, tracks_batch, frames_meta)
        ):
            frame_alerts = []
            
            # Rule 1: Suspicious object detection
            if self.rules["suspicious_object"]["enabled"]:
                suspicious_alerts = self._check_suspicious_objects(
                    classes, bboxes, confs, meta
                )
                frame_alerts.extend(suspicious_alerts)
            
            # Rule 2: Crowding detection
            if self.rules["crowding"]["enabled"]:
                crowding_alert = self._check_crowding(classes, meta)
                if crowding_alert:
                    frame_alerts.append(crowding_alert)
            
            # Rule 3: Loitering detection (requires tracks)
            if self.rules["loitering"]["enabled"] and tracks:
                loitering_alerts = self._check_loitering(tracks, meta)
                frame_alerts.extend(loitering_alerts)
            
            alerts.extend(frame_alerts)
        
        # Deduplicate alerts (remove duplicates within short time windows)
        alerts = self._deduplicate_alerts(alerts)
        
        print(f"Generated {len(alerts)} alerts")
        
        # Store metadata
        self.metadata = {
            "session_id": self.session_id,
            "alerts": alerts
        }
        
        return (alerts,)
    
    def _build_track_history(self, tracks_batch: TRACKS_BATCH, frames_meta: FRAME_META_BATCH):
        """Build history of track positions over time"""
        self.track_history.clear()
        
        for tracks, meta in zip(tracks_batch, frames_meta):
            timestamp = meta['timestamp_sec']
            
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                
                # Calculate center point
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                self.track_history[track_id].append({
                    'timestamp': timestamp,
                    'center': (center_x, center_y),
                    'bbox': bbox,
                    'cls': track['cls']
                })
    
    def _check_suspicious_objects(self, classes: List[str], bboxes: List[List[int]], 
                                   confs: List[float], meta: Dict) -> List[Dict]:
        """Check for suspicious/dangerous objects"""
        alerts = []
        watchlist = self.rules["suspicious_object"]["watchlist"]
        conf_threshold = self.rules["suspicious_object"]["conf_threshold"]
        
        for cls, bbox, conf in zip(classes, bboxes, confs):
            if cls in watchlist and conf >= conf_threshold:
                alert = {
                    "alert_id": str(uuid.uuid4()),
                    "timestamp_sec": meta['timestamp_sec'],
                    "frame_ref": f"frame_{meta['frame_index']:06d}",
                    "reason": f"suspicious_object_{cls}",
                    "evidence": {
                        "object_class": cls,
                        "confidence": conf,
                        "bbox": bbox
                    },
                    "severity": self.rules["suspicious_object"]["severity"]
                }
                alerts.append(alert)
        
        return alerts
    
    def _check_crowding(self, classes: List[str], meta: Dict) -> Dict:
        """Check for crowding (too many people)"""
        person_count = classes.count('person')
        threshold = self.rules["crowding"]["person_threshold"]
        
        if person_count >= threshold:
            return {
                "alert_id": str(uuid.uuid4()),
                "timestamp_sec": meta['timestamp_sec'],
                "frame_ref": f"frame_{meta['frame_index']:06d}",
                "reason": "crowding",
                "evidence": {
                    "person_count": person_count,
                    "threshold": threshold
                },
                "severity": self.rules["crowding"]["severity"]
            }
        
        return None
    
    def _check_loitering(self, tracks: List[Dict], meta: Dict) -> List[Dict]:
        """Check for loitering (person staying in small area for too long)"""
        alerts = []
        duration_threshold = self.rules["loitering"]["duration_threshold"]
        current_time = meta['timestamp_sec']
        
        for track in tracks:
            track_id = track['track_id']
            
            # Skip non-person tracks
            if track['cls'] != 'person':
                continue
            
            # Get track history
            history = self.track_history.get(track_id, [])
            
            if len(history) < 2:
                continue
            
            # Calculate duration
            first_time = history[0]['timestamp']
            duration = current_time - first_time
            
            if duration < duration_threshold:
                continue
            
            # Check if person stayed in small area (simple distance check)
            first_pos = history[0]['center']
            current_pos = history[-1]['center']
            
            distance = ((current_pos[0] - first_pos[0])**2 + 
                       (current_pos[1] - first_pos[1])**2)**0.5
            
            # If moved less than 100 pixels over the duration, it's loitering
            if distance < 100:
                alert = {
                    "alert_id": str(uuid.uuid4()),
                    "timestamp_sec": meta['timestamp_sec'],
                    "frame_ref": f"frame_{meta['frame_index']:06d}",
                    "reason": "loitering",
                    "evidence": {
                        "track_id": track_id,
                        "duration_sec": duration
                    },
                    "severity": self.rules["loitering"]["severity"]
                }
                alerts.append(alert)
        
        return alerts
    
    def _deduplicate_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Remove duplicate alerts within short time windows"""
        if not alerts:
            return alerts
        
        # Sort by timestamp
        sorted_alerts = sorted(alerts, key=lambda x: x['timestamp_sec'])
        
        deduplicated = []
        last_alert_by_reason = {}
        
        for alert in sorted_alerts:
            reason = alert['reason']
            timestamp = alert['timestamp_sec']
            
            # Check if similar alert was recently generated
            if reason in last_alert_by_reason:
                last_timestamp = last_alert_by_reason[reason]
                
                # If within 5 seconds, skip (duplicate)
                if timestamp - last_timestamp < 5.0:
                    continue
            
            deduplicated.append(alert)
            last_alert_by_reason[reason] = timestamp
        
        return deduplicated
    
    def _write_metadata(self, metadata_dir: Path) -> None:
        """Write alert.json"""
        alert_path = metadata_dir / "alert.json"
        self._save_json(alert_path, self.metadata)
        print(f"Saved alerts to {alert_path}")
