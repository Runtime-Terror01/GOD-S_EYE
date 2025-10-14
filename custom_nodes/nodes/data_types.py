"""
Standardized Data Types for Node-to-Node Transfer
All nodes must use these standardized types for interoperability
"""

from typing import List, Dict, Any, Union
import numpy as np
from dataclasses import dataclass

# ==================== Core Data Types ====================

# VIDEO: str (path) or an OpenCV VideoCapture object
VIDEO = Union[str, Any]  # str path or cv2.VideoCapture

# FRAMES_BATCH: List[np.ndarray] or a single np.ndarray shaped (N, H, W, C)
FRAMES_BATCH = Union[List[np.ndarray], np.ndarray]

# FRAME_META_BATCH: List[dict] (corresponding frames_metadata entries)
# Each dict must have: session_id, frame_id, frame_index, timestamp_sec, path
FRAME_META_BATCH = List[Dict[str, Any]]

# CLASS_BATCH: List[List[str]] (per frame)
# Example: [["person", "car"], ["person"], ["person", "person", "dog"]]
CLASS_BATCH = List[List[str]]

# BBOX_BATCH: List[List[List[int]]] (per frame list of [x1,y1,x2,y2])
# Example: [[[100,150,230,400], [300,200,500,450]], [[120,160,240,410]], ...]
BBOX_BATCH = List[List[List[int]]]

# CONF_BATCH: List[List[float]] (confidence scores per detection per frame)
# Example: [[0.92, 0.87], [0.95], [0.88, 0.91, 0.76]]
CONF_BATCH = List[List[float]]

# TRACKS_BATCH: List[List[dict]] where each dict is:
# {"track_id": int, "cls": str, "bbox": [x1,y1,x2,y2], "conf": float}
# Example: [
#   [{"track_id": 1, "cls": "person", "bbox": [100,150,230,400], "conf": 0.92}],
#   [{"track_id": 1, "cls": "person", "bbox": [105,152,235,402], "conf": 0.93}],
# ]
TRACKS_BATCH = List[List[Dict[str, Any]]]

# REPORT: dict (as report.json structure)
# Must contain: session_id, comprehensive_report, summary_for_user, generated_at
@dataclass
class ReportDict:
    session_id: str
    comprehensive_report: str
    summary_for_user: Dict[str, Any]  # overall_risk_level, justification, recommended_actions
    generated_at: str

REPORT = Dict[str, Any]

# ALERTS: List[dict] (as alert.json entries)
# Each dict must have: alert_id, timestamp_sec, frame_ref, reason, evidence, severity
@dataclass
class AlertDict:
    alert_id: str
    timestamp_sec: float
    frame_ref: str
    reason: str
    evidence: Dict[str, Any]
    severity: str  # "Low", "Medium", "High", "Critical"

ALERTS = List[Dict[str, Any]]


# ==================== Helper Functions ====================

def validate_frame_meta_batch(frame_meta_batch: FRAME_META_BATCH) -> bool:
    """Validate frame metadata batch structure"""
    if not isinstance(frame_meta_batch, list):
        return False
    
    required_keys = {"session_id", "frame_id", "frame_index", "timestamp_sec", "path"}
    for meta in frame_meta_batch:
        if not isinstance(meta, dict):
            return False
        if not required_keys.issubset(meta.keys()):
            return False
    
    return True


def validate_tracks_batch(tracks_batch: TRACKS_BATCH) -> bool:
    """Validate tracks batch structure"""
    if not isinstance(tracks_batch, list):
        return False
    
    required_keys = {"track_id", "cls", "bbox", "conf"}
    for frame_tracks in tracks_batch:
        if not isinstance(frame_tracks, list):
            return False
        for track in frame_tracks:
            if not isinstance(track, dict):
                return False
            if not required_keys.issubset(track.keys()):
                return False
            if not isinstance(track["bbox"], list) or len(track["bbox"]) != 4:
                return False
    
    return True


def validate_report(report: REPORT) -> bool:
    """Validate report structure"""
    if not isinstance(report, dict):
        return False
    
    required_keys = {"session_id", "comprehensive_report", "summary_for_user", "generated_at"}
    if not required_keys.issubset(report.keys()):
        return False
    
    summary_keys = {"overall_risk_level", "justification", "recommended_actions"}
    if not isinstance(report["summary_for_user"], dict):
        return False
    if not summary_keys.issubset(report["summary_for_user"].keys()):
        return False
    
    return True


def validate_alerts(alerts: ALERTS) -> bool:
    """Validate alerts structure"""
    if not isinstance(alerts, list):
        return False
    
    required_keys = {"alert_id", "timestamp_sec", "frame_ref", "reason", "evidence", "severity"}
    for alert in alerts:
        if not isinstance(alert, dict):
            return False
        if not required_keys.issubset(alert.keys()):
            return False
        if alert["severity"] not in ["Low", "Medium", "High", "Critical"]:
            return False
    
    return True
