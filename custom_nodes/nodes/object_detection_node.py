"""
Object Detection Node using YOLOv8
Detects objects in frames and outputs standardized detection data
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone

from .base_node import BaseNode
from .data_types import FRAMES_BATCH, FRAME_META_BATCH, CLASS_BATCH, BBOX_BATCH, CONF_BATCH

# Try to import YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Using mock detector.")


class ObjectDetectionNode(BaseNode):
    """
    Detect objects in frames using YOLOv8
    Outputs CLASS_BATCH, BBOX_BATCH, CONF_BATCH
    Saves detection_metadata.json and annotated frames
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.model = None
        self.session_id = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("FRAMES_BATCH",), 
                "frames_meta": ("FRAME_META_BATCH",),
            }
                }
            

    RETURN_TYPES = ("CLASS_BATCH", "BBOX_BATCH","CONF_BATCH")
    RETURN_NAMES = ("classes", "bboxes", "confidences")
    FUNCTION = "process"
    CATEGORY = "surveillance"
        
    def process(self, frames: FRAMES_BATCH, frames_meta: FRAME_META_BATCH) -> Tuple[CLASS_BATCH, BBOX_BATCH, CONF_BATCH]:
        """
        Run object detection on frames and save detection metadata & annotated frames
        """
        # Initialize model
        if self.model is None:
            self._load_model()

        # Extract session_id
        self.session_id = frames_meta[0]['session_id'] if frames_meta else "unknown"

        # Ensure frames is a list
        if isinstance(frames, np.ndarray):
            if len(frames.shape) == 4:  # (N, H, W, C)
                frames = [frames[i] for i in range(frames.shape[0])]
            else:
                frames = [frames]

        # Run detection on each frame
        class_batch, bbox_batch, conf_batch = [], [], []
        detections_dict = {}

        print(f"Running detection on {len(frames)} frames...")

        for idx, (frame, meta) in enumerate(zip(frames, frames_meta)):
            classes, bboxes, confs = self._detect_frame(frame)

            class_batch.append(classes)
            bbox_batch.append(bboxes)
            conf_batch.append(confs)

            # Store per-frame detections
            frame_key = f"frame_{meta['frame_index']:06d}"
            detections_dict[frame_key] = [
                {"cls": cls, "bbox": bbox, "conf": conf}
                for cls, bbox, conf in zip(classes, bboxes, confs)
            ]

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(frames)} frames")

        # Store metadata
        self.metadata = {
            'session_id': self.session_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'detections': detections_dict
        }

        # === SAVE METADATA AND ANNOTATED FRAMES ===
        output_dir = self.config.get('output_dir', 'surveillance_storage')
        session_path = Path(output_dir) / f"session_{self.session_id}"
        session_path.mkdir(parents=True, exist_ok=True)

        # Save metadata JSON
        self._write_metadata(session_path)

        # Save annotated frames
        self.save_annotated_frames(
            frames=frames,
            frames_meta=frames_meta,
            class_batch=class_batch,
            bbox_batch=bbox_batch,
            conf_batch=conf_batch,
            output_dir=output_dir
        )

        print(f"✅ Detection completed for session '{self.session_id}'")

        return class_batch, bbox_batch, conf_batch

    
    def _load_model(self):
        """Load YOLOv8 model or create mock detector"""
        model_path = self.config.get('model_path', 'yolov8n.pt')
        
        if YOLO_AVAILABLE:
            try:
                # Check if model file exists in current directory or use default
                if not os.path.exists(model_path):
                    model_path = 'yolov8n.pt'  # Will download automatically
                
                self.model = YOLO(model_path)
                print(f"Loaded YOLOv8 model: {model_path}")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                self.model = self._create_mock_detector()
        else:
            self.model = self._create_mock_detector()
    
    def _create_mock_detector(self):
        """Create a simple mock detector for testing"""
        class MockDetector:
            def predict(self, frame, conf=0.5, verbose=False):
                # Return mock detections
                h, w = frame.shape[:2]
                
                class MockResult:
                    def __init__(self):
                        self.boxes = MockBoxes()
                
                class MockBoxes:
                    def __init__(self):
                        # Generate 1-3 random detections
                        num_dets = np.random.randint(1, 4)
                        self.xyxy = []
                        self.conf = []
                        self.cls = []
                        
                        for _ in range(num_dets):
                            x1 = np.random.randint(0, w // 2)
                            y1 = np.random.randint(0, h // 2)
                            x2 = np.random.randint(x1 + 50, w)
                            y2 = np.random.randint(y1 + 50, h)
                            
                            self.xyxy.append([x1, y1, x2, y2])
                            self.conf.append(np.random.uniform(0.6, 0.95))
                            self.cls.append(np.random.choice([0, 1, 2]))  # person, car, etc.
                        
                        self.xyxy = np.array(self.xyxy) if self.xyxy else np.empty((0, 4))
                        self.conf = np.array(self.conf) if self.conf else np.empty(0)
                        self.cls = np.array(self.cls) if self.cls else np.empty(0)
                
                return [MockResult()]
        
        print("Using mock detector (install ultralytics for real detection)")
        return MockDetector()
    
    def _detect_frame(self, frame: np.ndarray) -> Tuple[List[str], List[List[int]], List[float]]:
        """
        Detect objects in a single frame
        
        Returns:
            (classes, bboxes, confidences)
        """
        conf_threshold = self.config.get('conf_threshold', 0.5)
        
        # Run inference
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        
        # Parse results
        classes = []
        bboxes = []
        confidences = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                # Get class names
                if YOLO_AVAILABLE and hasattr(self.model, 'names'):
                    class_names = self.model.names
                else:
                    # Mock class names
                    class_names = {0: 'person', 1: 'car', 2: 'dog'}
                
                # Extract detections
                for i in range(len(boxes.xyxy)):
                    bbox = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy[i], 'cpu') else boxes.xyxy[i]
                    conf = float(boxes.conf[i].cpu().numpy() if hasattr(boxes.conf[i], 'cpu') else boxes.conf[i])
                    cls_id = int(boxes.cls[i].cpu().numpy() if hasattr(boxes.cls[i], 'cpu') else boxes.cls[i])
                    
                    cls_name = class_names.get(cls_id, f'class_{cls_id}')
                    
                    classes.append(cls_name)
                    bboxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                    confidences.append(conf)
        
        return classes, bboxes, confidences
    
    def save_annotated_frames(self, frames: FRAMES_BATCH, frames_meta: FRAME_META_BATCH, 
                               class_batch: CLASS_BATCH, bbox_batch: BBOX_BATCH, 
                               conf_batch: CONF_BATCH, output_dir: str):
        """
        Save annotated frames with detection boxes
        
        Args:
            frames: Original frames
            frames_meta: Frame metadata
            class_batch: Detection classes
            bbox_batch: Detection bounding boxes
            conf_batch: Detection confidences
            output_dir: Base surveillance_storage directory
        """
        session_id = frames_meta[0]['session_id']
        storage_path = Path(output_dir) / f"session_{session_id}"
        annotated_dir = storage_path / "annotated_frame"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        for frame, meta, classes, bboxes, confs in zip(frames, frames_meta, class_batch, bbox_batch, conf_batch):
            # Draw boxes on frame
            annotated = frame.copy()
            
            for cls, bbox, conf in zip(classes, bboxes, confs):
                x1, y1, x2, y2 = bbox
                
                # Draw rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{cls}: {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save annotated frame
            frame_filename = f"frame_{meta['frame_index']:06d}.jpg"
            frame_path = annotated_dir / frame_filename
            cv2.imwrite(str(frame_path), annotated)
        
        print(f"Saved annotated frames to {annotated_dir}")
    
    def _write_metadata(self, metadata_dir: Path) -> None:
        """Write detection_metadata.json"""
        detection_meta_path = metadata_dir / "metadata" / "detection_metadata.json"
        self._save_json(detection_meta_path, self.metadata)
        print(f"Saved detection metadata to {detection_meta_path}")
