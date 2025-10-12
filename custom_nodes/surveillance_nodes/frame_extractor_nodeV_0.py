"""
Frame Extractor Node for ComfyUI
Extracts frames from video and saves them with proper metadata
"""

import os
import cv2
import uuid
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

class VideoFrameExtractorNode:
    """Extract frames from video and save with metadata"""
    
    def __init__(self):
        self.storage_root = Path("./surveillance_storage")
        self.frames_dir = self.storage_root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "extraction_mode": (["all_frames", "interval", "keyframes"], {
                    "default": "interval",
                    "tooltip": "Frame extraction strategy"
                }),
                "interval_seconds": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 60.0,
                    "step": 0.1,
                    "tooltip": "Extract frame every N seconds (for interval mode)"
                }),
                "max_frames": ("INT", {
                    "default": 300,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Maximum frames to extract"
                }),
                "quality": ("INT", {
                    "default": 85,
                    "min": 50,
                    "max": 100,
                    "tooltip": "JPEG quality for saved frames"
                }),
            },
            "optional": {
                "session_id": ("STRING", {
                    "default": "",
                    "tooltip": "Session ID for tracking (auto-generated if empty)"
                }),
            }
        }
    
    RETURN_TYPES = ("FRAME_METADATA",)
    RETURN_NAMES = ("frame_metadata",)
    FUNCTION = "extract_frames"
    CATEGORY = "Surveillance/Processing"
    
    def extract_frames(self, video, extraction_mode: str, interval_seconds: float, 
                      max_frames: int, quality: int, session_id: str = "") -> Tuple[Dict]:
        """
        Extract frames from video and save with metadata
        
        Returns:
            frame_metadata: Dictionary containing all frame information
        """
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get video path from ComfyUI video object
        video_path = video if isinstance(video, str) else video['video_path']
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Create session directory
        session_dir = self.frames_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Calculate frame extraction indices
        frame_indices = self._calculate_extraction_indices(
            extraction_mode, fps, total_frames, interval_seconds, max_frames
        )
        
        # Extract frames
        extracted_frames = []
        frames_in_memory = []
        
        print(f"Extracting {len(frame_indices)} frames from video...")
        
        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Failed to read frame {frame_idx}")
                continue
            
            # Generate frame ID and path
            frame_id = str(uuid.uuid4())
            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = session_dir / frame_filename
            
            # Save frame
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Calculate timestamp
            timestamp = frame_idx / fps if fps > 0 else 0
            
            # Create frame metadata
            frame_data = {
                'frame_id': frame_id,
                'frame_index': frame_idx,
                'extraction_index': idx,
                'timestamp': timestamp,
                'path': str(frame_path),
                'width': width,
                'height': height,
                'session_id': session_id,
                'metadata': {
                    'source_video': video_path,
                    'extraction_mode': extraction_mode,
                    'quality': quality
                }
            }
            
            extracted_frames.append(frame_data)
            frames_in_memory.append({
                'frame': frame,
                'metadata': frame_data
            })
        
        cap.release()
        
        # Create comprehensive metadata
        frame_metadata = {
            'session_id': session_id,
            'video_source': video_path,
            'storage_directory': str(session_dir),
            'extraction_timestamp': datetime.now().isoformat(),
            'video_properties': {
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration,
                'resolution': {
                    'width': width,
                    'height': height
                }
            },
            'extraction_settings': {
                'mode': extraction_mode,
                'interval_seconds': interval_seconds,
                'max_frames': max_frames,
                'quality': quality
            },
            'frames': extracted_frames,
            'frames_in_memory': frames_in_memory,  # For pipeline processing
            'total_frames_extracted': len(extracted_frames),
            'fps': fps,
            'duration': duration
        }
        
        # Save metadata to JSON
        metadata_path = session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            # Create a copy without frames_in_memory for JSON serialization
            json_metadata = {k: v for k, v in frame_metadata.items() if k != 'frames_in_memory'}
            json.dump(json_metadata, f, indent=2)
        
        print(f"✓ Extracted {len(extracted_frames)} frames to {session_dir}")
        print(f"✓ Session ID: {session_id}")
        
        return (frame_metadata,)
    
    def _calculate_extraction_indices(self, mode: str, fps: float, total_frames: int,
                                     interval_seconds: float, max_frames: int) -> List[int]:
        """Calculate which frames to extract based on mode"""
        
        if mode == "all_frames":
            # Extract all frames up to max_frames
            step = max(1, total_frames // max_frames) if total_frames > max_frames else 1
            indices = list(range(0, total_frames, step))[:max_frames]
            
        elif mode == "interval":
            # Extract frames at regular time intervals
            interval_frames = int(fps * interval_seconds)
            indices = list(range(0, total_frames, interval_frames))[:max_frames]
            
        elif mode == "keyframes":
            # Simple keyframe detection (every scene change)
            # For now, use uniform sampling as placeholder
            # TODO: Implement proper keyframe detection
            step = max(1, total_frames // min(max_frames, total_frames // 10))
            indices = list(range(0, total_frames, step))[:max_frames]
            
        else:
            indices = [0]  # Fallback to first frame
        
        return indices

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VideoFrameExtractorNode": VideoFrameExtractorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameExtractorNode": "Extract Video Frames"
}