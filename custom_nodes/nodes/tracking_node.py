"""
Tracking Node using ByteTrack
Tracks detected objects across frames to maintain identity
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from .base_node import BaseNode
from .data_types import CLASS_BATCH, BBOX_BATCH, CONF_BATCH, FRAME_META_BATCH, TRACKS_BATCH

# Try to import ByteTrack
try:
    from bytetrack.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("Warning: ByteTrack not available. Using simple IoU tracker.")


class SimpleIoUTracker:
    """Simple IoU-based tracker as fallback"""
    
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
        
    def update(self, bboxes: List[List[int]], scores: List[float], classes: List[str]) -> List[Dict]:
        """
        Update tracks with new detections
        
        Returns:
            List of track dictionaries
        """
        if not bboxes:
            # Age out tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return []
        
        # Match detections to existing tracks
        matched_tracks = []
        unmatched_dets = list(range(len(bboxes)))
        
        for track_id, track in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = -1
            
            for det_idx in unmatched_dets:
                iou = self._compute_iou(track['bbox'], bboxes[det_idx])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                # Update track
                self.tracks[track_id]['bbox'] = bboxes[best_det_idx]
                self.tracks[track_id]['cls'] = classes[best_det_idx]
                self.tracks[track_id]['conf'] = scores[best_det_idx]
                self.tracks[track_id]['age'] = 0
                
                matched_tracks.append({
                    'track_id': track_id,
                    'cls': classes[best_det_idx],
                    'bbox': bboxes[best_det_idx],
                    'conf': scores[best_det_idx]
                })
                
                unmatched_dets.remove(best_det_idx)
            else:
                track['age'] += 1
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = {
                'bbox': bboxes[det_idx],
                'cls': classes[det_idx],
                'conf': scores[det_idx],
                'age': 0
            }
            
            matched_tracks.append({
                'track_id': track_id,
                'cls': classes[det_idx],
                'bbox': bboxes[det_idx],
                'conf': scores[det_idx]
            })
        
        # Remove aged out tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        return matched_tracks
    
    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou


class TrackingNode(BaseNode):
    """
    Track objects across frames using ByteTrack or simple IoU tracker
    Outputs TRACKS_BATCH
    Saves tracking_metadata.json
    """
    
    # ComfyUI Node Configuration
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "classes": ("CLASS_BATCH",),
                "bboxes": ("BBOX_BATCH",),
                "confidences": ("CONF_BATCH",),
                "frame_metadata": ("FRAME_META_BATCH",),
                "iou_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.9, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("TRACKS_BATCH",)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "track"
    CATEGORY = "surveillance"
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.tracker = None
        self.session_id = None
    
    def track(self, classes, bboxes, confidences, frame_metadata, iou_threshold=0.3):
        """ComfyUI entry point"""
        self.config['iou_threshold'] = iou_threshold
        return self.process(classes, bboxes, confidences, frame_metadata)
        
    def process(self, class_batch: CLASS_BATCH, bbox_batch: BBOX_BATCH, 
                conf_batch: CONF_BATCH, frames_meta: FRAME_META_BATCH) -> Tuple[TRACKS_BATCH]:
        """
        Track objects across frames and save tracking metadata.
        """
        # Initialize tracker
        if self.tracker is None:
            self._initialize_tracker()

        # Extract session_id
        self.session_id = frames_meta[0]['session_id'] if frames_meta else "unknown"

        # Track objects frame by frame
        tracks_batch = []
        tracks_dict = {}

        print(f"Tracking objects across {len(class_batch)} frames...")

        for idx, (classes, bboxes, confs, meta) in enumerate(zip(class_batch, bbox_batch, conf_batch, frames_meta)):
            # Update tracker for this frame
            frame_tracks = self.tracker.update(bboxes, confs, classes)
            tracks_batch.append(frame_tracks)

            # Store formatted tracks for metadata
            frame_key = f"frame_{meta['frame_index']:06d}"
            tracks_dict[frame_key] = [
                {
                    "track_id": t["track_id"],
                    "cls": t["cls"],
                    "bbox": t["bbox"],
                    "conf": t["conf"]
                }
                for t in frame_tracks
            ]

            if (idx + 1) % 10 == 0:
                print(f"Tracked {idx + 1}/{len(class_batch)} frames")

        # === Store metadata in required format ===
        self.metadata = {
            "session_id": self.session_id,
            "tracks": tracks_dict
        }

        # === SAVE METADATA FILE ===
        output_dir = self.config.get("output_dir", "surveillance_storage")
        session_dir = Path(output_dir) / f"session_{self.session_id}"
        metadata_dir = session_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        self._write_metadata(metadata_dir)

        print(f"✅ Tracking metadata saved to {metadata_dir}/tracking_metadata.json")

        return (tracks_batch,)

    
    def _initialize_tracker(self):
        """Initialize ByteTrack or fallback tracker"""
        if BYTETRACK_AVAILABLE:
            try:
                # Try to use ByteTrack
                # Note: ByteTrack API may vary, adjust parameters as needed
                self.tracker = BYTETracker(
                    track_thresh=self.config.get('track_thresh', 0.5),
                    track_buffer=self.config.get('track_buffer', 30),
                    match_thresh=self.config.get('match_thresh', 0.8)
                )
                print("Using ByteTrack for tracking")
            except Exception as e:
                print(f"Error initializing ByteTrack: {e}")
                self._use_fallback_tracker()
        else:
            self._use_fallback_tracker()
    
    def _use_fallback_tracker(self):
        """Use simple IoU tracker as fallback"""
        self.tracker = SimpleIoUTracker(
            iou_threshold=self.config.get('iou_threshold', 0.3),
            max_age=self.config.get('max_age', 30)
        )
        print("Using simple IoU tracker")
    
    def _write_metadata(self, metadata_dir: Path) -> None:
        """Write tracking_metadata.json"""
        tracking_meta_path = metadata_dir / "tracking_metadata.json"
        self._save_json(tracking_meta_path, self.metadata)
        print(f"Saved tracking metadata to {tracking_meta_path}")
