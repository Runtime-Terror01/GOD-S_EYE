"""
Standardized Data Format Specification for Surveillance Pipeline
All nodes must follow these data structures for interoperability
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# ==================== Core Data Structures ====================

@dataclass
class SessionMetadata:
    """Unique session information for tracking data lineage"""
    session_id: str  # Unique session identifier
    pipeline_id: str  # Pipeline configuration ID
    timestamp: datetime  # Pipeline start time
    source_video: str  # Original video path/identifier
    
@dataclass
class FrameData:
    """Standard frame data structure"""
    frame_id: str  # Unique frame identifier (UUID)
    frame_index: int  # Sequential frame number
    timestamp: float  # Time in seconds from video start
    path: str  # Storage path for saved frame
    width: int
    height: int
    session_id: str  # Link to session
    metadata: Dict[str, Any] = None  # Additional metadata

@dataclass
class BoundingBox:
    """Standard bounding box format"""
    x1: float  # Top-left x (normalized 0-1)
    y1: float  # Top-left y (normalized 0-1)
    x2: float  # Bottom-right x (normalized 0-1)
    y2: float  # Bottom-right y (normalized 0-1)
    confidence: float  # Detection confidence (0-1)
    
@dataclass
class Detection:
    """Standard detection format"""
    detection_id: str  # Unique detection ID
    frame_id: str  # Reference to frame
    class_name: str  # Detected object class
    class_id: int  # Class ID from model
    bbox: BoundingBox  # Bounding box
    confidence: float  # Detection confidence
    attributes: Dict[str, Any] = None  # Additional attributes (color, pose, etc.)
    timestamp: float = None  # Frame timestamp
    
@dataclass
class Track:
    """Standard tracking data"""
    track_id: int  # Unique track ID
    detections: List[Detection]  # List of detections in track
    start_time: float  # Track start timestamp
    end_time: float  # Track end timestamp
    duration: float  # Track duration in seconds
    trajectory: List[tuple]  # List of (x, y, timestamp) points
    attributes: Dict[str, Any] = None  # Track attributes

@dataclass
class FaceData:
    """Face detection and recognition data"""
    face_id: str  # Unique face ID
    frame_id: str  # Reference to frame
    bbox: BoundingBox  # Face bounding box
    embedding: List[float]  # Face embedding vector
    identity: Optional[str] = None  # Recognized person ID
    confidence: Optional[float] = None  # Recognition confidence
    landmarks: Optional[Dict[str, tuple]] = None  # Facial landmarks

class ThreatLevel(Enum):
    """Standard threat levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ThreatAnalysis:
    """Threat analysis results"""
    threat_level: ThreatLevel
    threats_detected: List[str]  # List of threat types
    confidence: float  # Overall confidence
    detailed_analysis: Dict[str, Any]  # Detailed threat information
    timestamp: datetime
    frame_range: tuple  # (start_frame_id, end_frame_id)

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str  # Unique alert ID
    alert_type: str  # Type of alert
    threat_level: ThreatLevel
    timestamp: datetime
    location: Optional[str] = None  # Camera/zone location
    description: str = ""
    evidence: Dict[str, Any] = None  # Supporting evidence
    frame_ids: List[str] = None  # Related frames
    track_ids: List[int] = None  # Related tracks
    requires_action: bool = True
    assigned_to: Optional[str] = None

@dataclass
class VLMReport:
    """VLM analysis report structure"""
    report_id: str
    session_id: str
    timestamp: datetime
    comprehensive_report: str  # Detailed timestamped report
    executive_summary: str  # Brief summary
    risk_assessment: ThreatLevel
    key_events: List[Dict[str, Any]]  # List of important events
    recommendations: List[str]  # Recommended actions

# ==================== Inter-Node Message Format ====================

class NodeMessage:
    """Standard message format for inter-node communication"""
    
    def __init__(self, 
                 source_node: str,
                 target_node: str,
                 data_type: str,
                 data: Any,
                 session_metadata: SessionMetadata):
        self.message_id = str(uuid.uuid4())
        self.source_node = source_node
        self.target_node = target_node
        self.data_type = data_type  # 'frames', 'detections', 'tracks', etc.
        self.data = data
        self.session_metadata = session_metadata
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict:
        """Convert message to dictionary for serialization"""
        return {
            'message_id': self.message_id,
            'source_node': self.source_node,
            'target_node': self.target_node,
            'data_type': self.data_type,
            'data': self._serialize_data(self.data),
            'session_metadata': self._serialize_metadata(self.session_metadata),
            'timestamp': self.timestamp.isoformat()
        }
    
    def _serialize_data(self, data):
        """Serialize data based on type"""
        if isinstance(data, list):
            return [self._serialize_item(item) for item in data]
        return self._serialize_item(data)
    
    def _serialize_item(self, item):
        """Serialize individual data items"""
        if hasattr(item, '__dict__'):
            return {k: v for k, v in item.__dict__.items() if not k.startswith('_')}
        return item
    
    def _serialize_metadata(self, metadata):
        """Serialize session metadata"""
        return {
            'session_id': metadata.session_id,
            'pipeline_id': metadata.pipeline_id,
            'timestamp': metadata.timestamp.isoformat(),
            'source_video': metadata.source_video
        }

# ==================== Database Schema Reference ====================

DATABASE_SCHEMAS = {
    "frames": {
        "collection": "frames",
        "indexes": ["frame_id", "session_id", "timestamp"],
        "schema": {
            "frame_id": "string",
            "session_id": "string",
            "frame_index": "integer",
            "timestamp": "float",
            "path": "string",
            "width": "integer",
            "height": "integer",
            "metadata": "object"
        }
    },
    "detections": {
        "collection": "detections",
        "indexes": ["detection_id", "frame_id", "class_name"],
        "schema": {
            "detection_id": "string",
            "frame_id": "string",
            "class_name": "string",
            "bbox": "object",
            "confidence": "float",
            "timestamp": "float"
        }
    },
    "tracks": {
        "collection": "tracks",
        "indexes": ["track_id", "start_time", "end_time"],
        "schema": {
            "track_id": "integer",
            "detections": "array",
            "trajectory": "array",
            "duration": "float"
        }
    },
    "alerts": {
        "collection": "alerts",
        "indexes": ["alert_id", "timestamp", "threat_level"],
        "schema": {
            "alert_id": "string",
            "alert_type": "string",
            "threat_level": "string",
            "timestamp": "datetime",
            "evidence": "object"
        }
    },
    "vlm_reports": {
        "collection": "vlm_reports",
        "indexes": ["report_id", "session_id", "timestamp"],
        "schema": {
            "report_id": "string",
            "session_id": "string",
            "comprehensive_report": "text",
            "executive_summary": "text",
            "risk_assessment": "string"
        }
    }
}

# ==================== Vector Database Schema ====================

VECTOR_SCHEMAS = {
    "scene_embeddings": {
        "collection": "scene_embeddings",
        "vector_dimension": 768,  # Based on model used
        "metadata_fields": ["session_id", "timestamp", "frame_id", "description"],
        "index_type": "HNSW"  # Hierarchical Navigable Small World
    },
    "face_embeddings": {
        "collection": "face_embeddings", 
        "vector_dimension": 512,  # Standard face embedding size
        "metadata_fields": ["face_id", "identity", "timestamp", "frame_id"],
        "index_type": "IVF_FLAT"  # Inverted File Index
    }
}

import uuid