"""
Threat Detection Node
RF-DETR based threat detection for images
"""

import torch
import numpy as np
import cv2
from PIL import Image
import time
import json
import uuid
import os

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("Warning: supervision library not available. Install with: pip install supervision==0.26.1")

class ThreatDetectionNode:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("THREAT_MODEL",),
                "frame_metadata": ("FRAME_METADATA",),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.9,
                    "step": 0.05,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("predictions_json",)
    FUNCTION = "detect_threats"
    CATEGORY = "Threat Detection"
    
    # # COMMENTED OUT: Annotation functions no longer needed for JSON output
    # def setup_annotators(self, threat_classes):
    #     """Setup supervision annotators with threat-specific colors"""
    #     if not SUPERVISION_AVAILABLE:
    #         return None, None
    #         
    #     # Define colors for each threat class
    #     color_palette = sv.ColorPalette.from_hex([
    #         "#FF0000",  # Gun - Red
    #         "#FF8C00",  # Explosive - Orange  
    #         "#FF4500",  # Grenade - Red-Orange
    #         "#DC143C"   # Knife - Crimson
    #     ])
    #     
    #     bbox_annotator = sv.BoxAnnotator(
    #         color=color_palette, 
    #         thickness=3
    #     )
    #     
    #     label_annotator = sv.LabelAnnotator(
    #         color=color_palette,
    #         text_color=sv.Color.WHITE,
    #         text_scale=1.0,
    #         text_thickness=2,
    #         smart_position=True
    #     )
    #     
    #     return bbox_annotator, label_annotator
    
    def process_single_image(self, pil_image, model, threshold, threat_classes):
        """Process a single PIL image for threat detection"""
        
        # Run inference
        start_time = time.time()
        detections = model.predict(pil_image, threshold=threshold)
        inference_time = time.time() - start_time
        
        # Get image dimensions
        width, height = pil_image.size
        
        # Create prediction data in the requested format
        predictions = []
        if len(detections) > 0:
            for bbox, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
                x1, y1, x2, y2 = bbox
                
                # Calculate center point and dimensions
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Get class name
                class_name = threat_classes.get(class_id, f"unknown_class_{class_id}")
                
                prediction = {
                    'x': float(center_x),
                    'y': float(center_y),
                    'width': float(bbox_width),
                    'height': float(bbox_height),
                    'confidence': float(confidence),
                    'class': class_name.lower(),
                    'class_id': int(class_id),
                    'detection_id': str(uuid.uuid4())
                }
                predictions.append(prediction)
        
        # Create the full prediction result
        result = {
            'inference_id': str(uuid.uuid4()),
            'time': inference_time,
            'image': {
                'width': width,
                'height': height
            },
            'predictions': predictions
        }
        
        return result
        
        # # COMMENTED OUT: Original annotation logic
        # # Create labels if we have detections
        # labels = []
        # if len(detections) > 0:
        #     for class_id, confidence in zip(detections.class_id, detections.confidence):
        #         class_name = threat_classes.get(class_id, f"unknown_class_{class_id}")
        #         labels.append(f"{class_name} {confidence:.2f}")
        # 
        # # Annotate image if supervision is available
        # if SUPERVISION_AVAILABLE and bbox_annotator and label_annotator:
        #     annotated_pil = pil_image.copy()
        #     if len(detections) > 0:
        #         annotated_pil = bbox_annotator.annotate(annotated_pil, detections)
        #         annotated_pil = label_annotator.annotate(annotated_pil, detections, labels)
        # else:
        #     # Fallback: basic OpenCV annotation
        #     annotated_pil = self.annotate_with_opencv(pil_image, detections, labels, threat_classes)
        # 
        # return annotated_pil, detections, labels
    
    # # COMMENTED OUT: OpenCV annotation function no longer needed for JSON output
    # def annotate_with_opencv(self, pil_image, detections, labels, threat_classes):
    #     """Fallback annotation using OpenCV when supervision is not available"""
    #     
    #     # Convert PIL to OpenCV format
    #     cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    #     
    #     if len(detections) > 0:
    #         # Define colors for each threat class (BGR format for OpenCV)
    #         colors = {
    #             1: (0, 0, 255),    # Gun - Red
    #             2: (0, 140, 255),  # Explosive - Orange
    #             3: (0, 69, 255),   # Grenade - Red-Orange  
    #             4: (60, 20, 220)   # Knife - Crimson
    #         }
    #         
    #         for i, (bbox, class_id, confidence) in enumerate(zip(
    #             detections.xyxy, detections.class_id, detections.confidence
    #         )):
    #             x1, y1, x2, y2 = bbox.astype(int)
    #             color = colors.get(class_id, (255, 255, 255))
    #             
    #             # Draw bounding box
    #             cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 3)
    #             
    #             # Draw label
    #             if i < len(labels):
    #                 label = labels[i]
    #                 (text_width, text_height), baseline = cv2.getTextSize(
    #                     label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    #                 )
    #                 
    #                 # Draw label background
    #                 cv2.rectangle(cv_image, 
    #                             (x1, y1 - text_height - baseline - 5),
    #                             (x1 + text_width, y1),
    #                             color, -1)
    #                 
    #                 # Draw label text
    #                 cv2.putText(cv_image, label, (x1, y1 - 5),
    #                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #     
    #     # Convert back to PIL
    #     annotated_pil = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    #     return annotated_pil
    
    def process_image_batch(self, frames, model, threshold, threat_classes, device):
        """Process a batch of frames efficiently and return JSON predictions"""
        
        total_frames = len(frames)
        all_predictions = []
        total_detections = 0
        
        print(f"Processing {total_frames} images on {device}...")
        
        batch_start_time = time.time()
        
        # Process each frame individually
        for i, frame in enumerate(frames):
            prediction_result = self.process_single_image(
                frame, model, threshold, threat_classes
            )
            all_predictions.append(prediction_result)
            total_detections += len(prediction_result['predictions'])
            
            # Progress update every 10 frames
            if (i + 1) % 10 == 0:
                elapsed_time = time.time() - batch_start_time
                fps = (i + 1) / elapsed_time if elapsed_time > 0 else 0
                print(f"Processed {i + 1}/{total_frames} images, {fps:.1f} FPS")
            
            # Clear GPU cache periodically for GPU processing
            if device == "cuda" and i % 50 == 0:
                torch.cuda.empty_cache()
        
        # Final progress update
        total_time = time.time() - batch_start_time
        avg_fps = total_frames / total_time if total_time > 0 else 0
        
        print(f"✓ Processing complete: {total_frames} images processed, "
              f"{avg_fps:.1f} avg FPS, {total_detections} total threats detected")
        
        return all_predictions
    
    def process_frames_from_metadata(self, frame_metadata, model, threshold, threat_classes, device):
        """Process frames using metadata references - load frames on demand"""
        
        frames_info = frame_metadata['frames']
        total_frames = len(frames_info)
        all_predictions = []
        total_detections = 0
        
        print(f"Processing {total_frames} frames from storage on {device}...")
        
        batch_start_time = time.time()
        
        # Process each frame by loading from disk
        for i, frame_info in enumerate(frames_info):
            # Load frame from disk
            frame_path = frame_info['path']
            if not os.path.exists(frame_path):
                print(f"Warning: Frame file not found: {frame_path}")
                continue
            
            # Load image
            pil_image = Image.open(frame_path)
            
            # Run inference
            start_time = time.time()
            detections = model.predict(pil_image, threshold=threshold)
            inference_time = time.time() - start_time
            
            # Create prediction data with frame reference
            predictions = []
            if len(detections) > 0:
                for bbox, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
                    x1, y1, x2, y2 = bbox
                    
                    # Calculate center point and dimensions
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # Get class name
                    class_name = threat_classes.get(class_id, f"unknown_class_{class_id}")
                    
                    prediction = {
                        'x': float(center_x),
                        'y': float(center_y),
                        'width': float(bbox_width),
                        'height': float(bbox_height),
                        'confidence': float(confidence),
                        'class': class_name.lower(),
                        'class_id': int(class_id),
                        'detection_id': str(uuid.uuid4())
                    }
                    predictions.append(prediction)
            
            # Create the full prediction result with frame reference
            result = {
                'inference_id': str(uuid.uuid4()),
                'time': inference_time,
                'frame_reference': {
                    'frame_id': frame_info['frame_id'],
                    'frame_index': frame_info['frame_index'],
                    'timestamp': frame_info['timestamp'],
                    'path': frame_info['path']
                },
                'image': {
                    'width': frame_info['width'],
                    'height': frame_info['height']
                },
                'predictions': predictions
            }
            
            all_predictions.append(result)
            total_detections += len(predictions)
            
            # Progress update every 10 frames
            if (i + 1) % 10 == 0:
                elapsed_time = time.time() - batch_start_time
                fps = (i + 1) / elapsed_time if elapsed_time > 0 else 0
                print(f"Processed {i + 1}/{total_frames} frames, {fps:.1f} FPS")
            
            # Clear GPU cache periodically for GPU processing
            if device == "cuda" and i % 50 == 0:
                torch.cuda.empty_cache()
        
        # Final progress update
        total_time = time.time() - batch_start_time
        avg_fps = total_frames / total_time if total_time > 0 else 0
        
        print(f"✓ Processing complete: {total_frames} frames processed, "
              f"{avg_fps:.1f} avg FPS, {total_detections} total threats detected")
        
        return all_predictions
        
        # # COMMENTED OUT: Original batch processing logic
        # for i in range(0, total_frames, batch_size):
        #     batch_end = min(i + batch_size, total_frames)
        #     batch_frames = frames[i:batch_end]
        #     
        #     batch_start_time = time.time()
        #     batch_annotated = []
        #     batch_detections = 0
        #     
        #     # Process each frame in the batch
        #     for frame in batch_frames:
        #         annotated_frame, detections, labels = self.process_single_image(
        #             frame, model, threshold, threat_classes, bbox_annotator, label_annotator
        #         )
        #         batch_annotated.append(annotated_frame)
        #         batch_detections += len(detections)
        #         total_detections += len(detections)
        #     
        #     annotated_frames.extend(batch_annotated)
        #     
        #     # Progress update
        #     batch_time = time.time() - batch_start_time
        #     batch_fps = len(batch_frames) / batch_time if batch_time > 0 else 0
        #     
        #     print(f"Batch {i//batch_size + 1}: {len(batch_frames)} images, "
        #           f"{batch_fps:.1f} FPS, {batch_detections} threats detected")
        #     
        #     # Clear GPU cache periodically for GPU processing
        #     if device == "cuda" and (i // batch_size) % 5 == 0:
        #         torch.cuda.empty_cache()
        # 
        # print(f"✓ Processing complete: {total_detections} total threats detected")
        # return annotated_frames
    
    def detect_threats(self, model, frame_metadata, threshold):
        """Main threat detection function using frame metadata"""
        
        print(f"\n=== Threat Detection ===")
        print(f"Model: {model['model_name']}")
        print(f"Resolution: {model['resolution']}x{model['resolution']}")
        print(f"Device: {model['device']}")
        print(f"Threshold: {threshold}")
        
        # Get model components
        rf_model = model['model']
        threat_classes = model['threat_classes']
        device = model['device']
        
        # Process frame metadata
        if isinstance(frame_metadata, dict):
            print(f"Processing {frame_metadata['total_frames_extracted']} frames from metadata")
            print(f"Session ID: {frame_metadata['session_id']}")
            print(f"Video source: {frame_metadata['video_source']}")
            
            # Process each frame using metadata
            all_predictions = self.process_frames_from_metadata(
                frame_metadata, rf_model, threshold, threat_classes, device
            )
            
        else:
            raise ValueError(f"Unsupported frame_metadata format: {type(frame_metadata)}")
        
        # Convert predictions to JSON string
        predictions_json = json.dumps(all_predictions, indent=2)
        
        print(f"✓ Output: {len(all_predictions)} prediction results")
        
        return (predictions_json,)
        
        # # COMMENTED OUT: Original annotation logic
        # # Setup annotators
        # bbox_annotator, label_annotator = self.setup_annotators(threat_classes)
        # 
        # # Process frames
        # annotated_pil_frames = self.process_image_batch(
        #     pil_frames, rf_model, threshold, threat_classes, 
        #     bbox_annotator, label_annotator, batch_size, device
        # )
        # 
        # # Convert annotated PIL frames back to ComfyUI tensor format
        # annotated_arrays = []
        # for pil_frame in annotated_pil_frames:
        #     frame_array = np.array(pil_frame).astype(np.float32) / 255.0
        #     annotated_arrays.append(frame_array)
        # 
        # # Stack into tensor [B, H, W, C]
        # output_tensor = torch.from_numpy(np.stack(annotated_arrays, axis=0))
        # 
        # print(f"✓ Output: {output_tensor.shape[0]} annotated images")
        # 
        # return (output_tensor,)

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ThreatDetectionNode": ThreatDetectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThreatDetectionNode": "Threat Detection"
}