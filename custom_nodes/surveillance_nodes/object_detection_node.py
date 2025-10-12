"""
Object Detection Node using YOLOv8
Detects objects in frames and outputs detailed detection data
"""

import os
import cv2
import uuid
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# YOLOv8 imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics YOLO not available. Install with: pip install ultralytics")

class ObjectDetectionNode:
    """Detect objects in frames using YOLOv8"""
    
    # Threat-related classes from COCO dataset
    THREAT_CLASSES = {
        'knife': 'weapon',
        'scissors': 'potential_weapon',
        'baseball bat': 'potential_weapon',
        'bottle': 'potential_weapon',
        'person': 'person',
        'car': 'vehicle',
        'truck': 'vehicle',
        'motorcycle': 'vehicle',
        'bicycle': 'vehicle',
        'backpack': 'suspicious_item',
        'handbag': 'suspicious_item',
        'suitcase': 'suspicious_item'
    }
    
    def __init__(self):
        self.model = None
        self.model_name = "yolov8n.pt"  # nano version for speed
        self.device = "cuda:0" if self._check_cuda() else "cpu"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_metadata": ("FRAME_METADATA",),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.95,
                    "step": 0.05,
                    "tooltip": "Minimum confidence for detections"
                }),
                "nms_threshold": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "tooltip": "Non-max suppression threshold"
                }),
                "detect_classes": ("STRING", {
                    "default": "all",
                    "tooltip": "Classes to detect (comma-separated) or 'all'"
                }),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Custom YOLO model path (uses YOLOv8n if empty)"
                }),
                "save_annotated": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save annotated frames"
                }),
            }
        }
    
    RETURN_TYPES = ("DETECTIONS", "DETECTION_METADATA")
    RETURN_NAMES = ("detections", "detection_metadata")
    FUNCTION = "detect_objects"
    CATEGORY = "Surveillance/Detection"
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _load_model(self, model_path: str = "") -> YOLO:
        """Load YOLO model"""
        if not YOLO_AVAILABLE:
            raise RuntimeError("YOLOv8 not available. Install with: pip install ultralytics")
        
        if model_path and os.path.exists(model_path):
            print(f"Loading custom model: {model_path}")
            return YOLO(model_path)
        else:
            print(f"Loading default model: {self.model_name}")
            return YOLO(self.model_name)
    
    def detect_objects(self, frame_metadata: Dict, confidence_threshold: float,
                      nms_threshold: float, detect_classes: str,
                      model_path: str = "", save_annotated: bool = True) -> Tuple[List, Dict]:
        """
        Perform object detection on frames
        
        Returns:
            detections: List of detection objects
            detection_metadata: Summary and statistics
        """
        
        # Load model if not already loaded
        if self.model is None:
            self.model = self._load_model(model_path)
            print(f"Model loaded on device: {self.device}")
        
        # Parse classes to detect
        if detect_classes.lower() == "all":
            target_classes = None  # Detect all classes
        else:
            target_classes = [c.strip() for c in detect_classes.split(',')]
        
        # Get frames from metadata
        frames_data = frame_metadata.get('frames_in_memory', [])
        if not frames_data:
            # Load frames from disk if not in memory
            frames_data = self._load_frames_from_disk(frame_metadata)
        
        session_id = frame_metadata['session_id']
        
        # Create directory for annotated frames if needed
        if save_annotated:
            annotated_dir = Path(frame_metadata['storage_directory']) / "annotated"
            annotated_dir.mkdir(exist_ok=True)
        
        # Process frames
        all_detections = []
        frame_detections_map = {}
        class_counts = {}
        threat_frames = set()
        
        print(f"Processing {len(frames_data)} frames for object detection...")
        
        for frame_info in frames_data:
            frame = frame_info['frame'] if 'frame' in frame_info else cv2.imread(frame_info['metadata']['path'])
            frame_meta = frame_info['metadata'] if 'metadata' in frame_info else frame_info
            
            # Run YOLO detection
            results = self.model(
                frame,
                conf=confidence_threshold,
                iou=nms_threshold,
                device=self.device,
                verbose=False
            )
            
            frame_detections = []
            
            # Process detections
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes
                    
                    for i in range(len(boxes)):
                        # Get detection info
                        box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        cls_name = self.model.names[cls_id]
                        
                        # Filter by target classes if specified
                        if target_classes and cls_name not in target_classes:
                            continue
                        
                        # Normalize coordinates
                        h, w = frame.shape[:2]
                        x1, y1, x2, y2 = box
                        norm_bbox = {
                            'x1': float(x1 / w),
                            'y1': float(y1 / h),
                            'x2': float(x2 / w),
                            'y2': float(y2 / h),
                            'confidence': conf
                        }
                        
                        # Check if it's a threat class
                        threat_category = self.THREAT_CLASSES.get(cls_name, None)
                        if threat_category:
                            threat_frames.add(frame_meta['frame_id'])
                        
                        # Create detection object
                        detection = {
                            'detection_id': str(uuid.uuid4()),
                            'frame_id': frame_meta['frame_id'],
                            'frame_index': frame_meta['frame_index'],
                            'timestamp': frame_meta['timestamp'],
                            'class_name': cls_name,
                            'class_id': cls_id,
                            'bbox': norm_bbox,
                            'confidence': conf,
                            'attributes': {
                                'threat_category': threat_category,
                                'pixel_bbox': {
                                    'x1': float(x1), 'y1': float(y1),
                                    'x2': float(x2), 'y2': float(y2)
                                },
                                'area': float((x2 - x1) * (y2 - y1)),
                                'center': {
                                    'x': float((x1 + x2) / 2),
                                    'y': float((y1 + y2) / 2)
                                }
                            }
                        }
                        
                        frame_detections.append(detection)
                        
                        # Update class counts
                        if cls_name not in class_counts:
                            class_counts[cls_name] = 0
                        class_counts[cls_name] += 1
            
            # Store frame detections
            if frame_detections:
                frame_detections_map[frame_meta['frame_id']] = frame_detections
                all_detections.extend(frame_detections)
            
            # Save annotated frame if requested
            if save_annotated and frame_detections:
                annotated_frame = self._draw_annotations(frame, frame_detections)
                annotated_path = annotated_dir / f"annotated_{frame_meta['frame_index']:06d}.jpg"
                cv2.imwrite(str(annotated_path), annotated_frame)
        
        # Create detection metadata summary
        detection_metadata = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'total_frames_processed': len(frames_data),
            'frames_with_detections': len(frame_detections_map),
            'total_detections': len(all_detections),
            'detection_settings': {
                'model': model_path if model_path else self.model_name,
                'confidence_threshold': confidence_threshold,
                'nms_threshold': nms_threshold,
                'target_classes': target_classes if target_classes else 'all',
                'device': self.device
            },
            'class_distribution': class_counts,
            'threat_analysis': {
                'threat_frames': list(threat_frames),
                'threat_frame_count': len(threat_frames),
                'threat_percentage': (len(threat_frames) / len(frames_data) * 100) if frames_data else 0,
                'detected_threats': [cls for cls in class_counts.keys() if cls in self.THREAT_CLASSES]
            },
            'frame_detections_map': frame_detections_map,
            'all_detections': all_detections
        }
        
        # Save metadata
        metadata_path = Path(frame_metadata['storage_directory']) / "detection_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({k: v for k, v in detection_metadata.items() 
                      if k not in ['all_detections', 'frame_detections_map']}, f, indent=2)
        
        print(f"✓ Detected {len(all_detections)} objects in {len(frame_detections_map)} frames")
        if class_counts:
            print(f"✓ Classes found: {', '.join(f'{k}:{v}' for k, v in class_counts.items())}")
        
        return (all_detections, detection_metadata)
    
    def _load_frames_from_disk(self, frame_metadata: Dict) -> List[Dict]:
        """Load frames from disk based on metadata"""
        frames_data = []
        for frame_info in frame_metadata.get('frames', []):
            frame_path = frame_info['path']
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                frames_data.append({
                    'frame': frame,
                    'metadata': frame_info
                })
        return frames_data
    
    def _draw_annotations(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        for det in detections:
            # Get pixel coordinates
            bbox = det['attributes']['pixel_bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Get color based on threat category
            threat_cat = det['attributes'].get('threat_category')
            if threat_cat == 'weapon':
                color = (0, 0, 255)  # Red for weapons
            elif threat_cat in ['potential_weapon', 'suspicious_item']:
                color = (0, 165, 255)  # Orange for suspicious
            elif threat_cat == 'person':
                color = (255, 0, 0)  # Blue for person
            else:
                color = (0, 255, 0)  # Green for normal
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_bg_end = (x1 + label_size[0] + 4, y1 - label_size[1] - 8)
            
            cv2.rectangle(annotated, (x1, y1 - 2), label_bg_end, color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ObjectDetectionNode": ObjectDetectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ObjectDetectionNode": "YOLO Object Detection"
}