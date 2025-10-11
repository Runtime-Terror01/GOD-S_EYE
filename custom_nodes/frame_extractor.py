"""
Video Frame Extractor Node
Extracts frames from video files and saves as JPG with metadata for efficient processing
"""

import torch
import numpy as np
import cv2
from PIL import Image
import os
import re
import time
import json
import uuid
from datetime import datetime

class VideoFrameExtractor:
    def __init__(self):
        # Create storage directory if it doesn't exist
        self.storage_dir = os.path.join(os.getcwd(), "storage")
        os.makedirs(self.storage_dir, exist_ok=True)
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "max_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Maximum frames to extract from video (0 = all frames). Use to limit processing time for long videos."
                }),
            }
        }
    
    RETURN_TYPES = ("FRAME_METADATA",)
    RETURN_NAMES = ("frame_metadata",)
    FUNCTION = "extract_frames"
    CATEGORY = "Video Processing"
    
    def extract_frames_to_storage(self, video_path, max_frames=None):
        """Extract frames from video and save as JPG files with metadata"""
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Create unique session ID and video-specific directory
        session_id = str(uuid.uuid4())[:8]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_storage_dir = os.path.join(self.storage_dir, f"{video_name}_{session_id}")
        os.makedirs(video_storage_dir, exist_ok=True)
        
        print(f"Extracting frames from video: {video_path}")
        print(f"Storage directory: {video_storage_dir}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s, {width}x{height}")
        
        # Limit frames if specified
        if max_frames and max_frames < total_frames:
            frame_step = max(1, total_frames // max_frames)
            print(f"Sampling every {frame_step} frames to limit to {max_frames} frames")
        else:
            frame_step = 1
            max_frames = total_frames
        
        frame_metadata_list = []
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on step
                if frame_count % frame_step == 0:
                    # Generate unique frame ID
                    frame_id = str(uuid.uuid4())
                    frame_filename = f"frame_{extracted_count:06d}_{frame_id[:8]}.jpg"
                    frame_path = os.path.join(video_storage_dir, frame_filename)
                    
                    # Calculate timestamp
                    timestamp = frame_count / fps if fps > 0 else 0
                    
                    # Convert BGR to RGB and save as JPG
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    pil_frame.save(frame_path, "JPEG", quality=95)
                    
                    # Create frame metadata
                    frame_metadata = {
                        "frame_id": frame_id,
                        "frame_index": extracted_count,
                        "original_frame_number": frame_count,
                        "timestamp": timestamp,
                        "path": frame_path,
                        "width": width,
                        "height": height,
                        "video_source": video_path,
                        "session_id": session_id
                    }
                    
                    frame_metadata_list.append(frame_metadata)
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
                
                # Progress update every 100 frames
                if frame_count % 100 == 0:
                    print(f"Extracted {extracted_count} frames from {frame_count} total frames...")
        
        finally:
            cap.release()
        
        # Create overall metadata
        overall_metadata = {
            "session_id": session_id,
            "video_source": video_path,
            "video_name": video_name,
            "storage_directory": video_storage_dir,
            "total_frames_extracted": len(frame_metadata_list),
            "original_total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "resolution": {"width": width, "height": height},
            "extraction_timestamp": datetime.now().isoformat(),
            "frames": frame_metadata_list
        }
        
        print(f"✓ Extracted {len(frame_metadata_list)} frames to storage")
        return overall_metadata

    def get_video_file_path(self, video_object):
        """Extract video file path from various ComfyUI video object types"""
        
        # Method 1: Try common file path attributes
        file_attrs = ['file', 'path', 'filename', '_file', 'video_path', 'filepath']
        for attr in file_attrs:
            if hasattr(video_object, attr):
                path = getattr(video_object, attr)
                if path and isinstance(path, str) and os.path.exists(path):
                    return path
        
        # Method 2: Try ComfyUI video object methods
        if hasattr(video_object, 'get_stream_source'):
            try:
                source = video_object.get_stream_source()
                if isinstance(source, str) and os.path.exists(source):
                    return source
            except:
                pass
        
        # Method 3: Try get_components method
        if hasattr(video_object, 'get_components'):
            try:
                components = video_object.get_components()
                if isinstance(components, dict):
                    for key in ['path', 'file', 'source', 'filename']:
                        if key in components and os.path.exists(components[key]):
                            return components[key]
                elif isinstance(components, (list, tuple)) and len(components) > 0:
                    # First component might be the file path
                    if isinstance(components[0], str) and os.path.exists(components[0]):
                        return components[0]
            except:
                pass
        
        # Method 4: Try save_to method to get file path
        if hasattr(video_object, 'save_to'):
            try:
                # Create a temporary path to see what the object would save
                temp_path = "temp_video_check.mp4"
                video_object.save_to(temp_path)
                if os.path.exists(temp_path):
                    # Get the actual source path if possible
                    actual_path = temp_path
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    return actual_path
            except:
                pass
        
        # Method 5: Parse string representation for file paths
        video_str = str(video_object)
        
        # Look for common video file extensions in the string
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
        
        for ext in video_extensions:
            # Find paths ending with video extensions
            pattern = r'([A-Za-z]:[\\\/][^<>:"|?*\n\r]+' + re.escape(ext) + r')'
            matches = re.findall(pattern, video_str, re.IGNORECASE)
            
            for match in matches:
                if os.path.exists(match):
                    return match
        
        return None

    def create_tensor_metadata(self, tensor, max_frames=0):
        """Create metadata for tensor-based video and save frames to storage"""
        
        # Limit frames if specified
        if max_frames > 0 and tensor.shape[0] > max_frames:
            indices = torch.linspace(0, tensor.shape[0] - 1, max_frames).long()
            tensor = tensor[indices]
            print(f"Sampled {max_frames} frames from {tensor.shape[0]} total frames")
        
        # Create unique session ID and directory
        session_id = str(uuid.uuid4())[:8]
        video_name = f"tensor_video_{session_id}"
        video_storage_dir = os.path.join(self.storage_dir, video_name)
        os.makedirs(video_storage_dir, exist_ok=True)
        
        print(f"Storage directory: {video_storage_dir}")
        
        # Convert tensor to numpy and save frames
        frames_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        frame_metadata_list = []
        
        for i, frame_np in enumerate(frames_np):
            # Generate unique frame ID
            frame_id = str(uuid.uuid4())
            frame_filename = f"frame_{i:06d}_{frame_id[:8]}.jpg"
            frame_path = os.path.join(video_storage_dir, frame_filename)
            
            # Save frame as JPG
            pil_frame = Image.fromarray(frame_np)
            pil_frame.save(frame_path, "JPEG", quality=95)
            
            # Create frame metadata
            frame_metadata = {
                "frame_id": frame_id,
                "frame_index": i,
                "original_frame_number": i,
                "timestamp": i * (1.0 / 30.0),  # Assume 30 FPS for tensor videos
                "path": frame_path,
                "width": frame_np.shape[1],
                "height": frame_np.shape[0],
                "video_source": "tensor_input",
                "session_id": session_id
            }
            
            frame_metadata_list.append(frame_metadata)
        
        # Create overall metadata
        overall_metadata = {
            "session_id": session_id,
            "video_source": "tensor_input",
            "video_name": video_name,
            "storage_directory": video_storage_dir,
            "total_frames_extracted": len(frame_metadata_list),
            "original_total_frames": tensor.shape[0],
            "fps": 30.0,  # Assume 30 FPS for tensor videos
            "duration": len(frame_metadata_list) / 30.0,
            "resolution": {"width": frame_np.shape[1], "height": frame_np.shape[0]},
            "extraction_timestamp": datetime.now().isoformat(),
            "frames": frame_metadata_list
        }
        
        print(f"✓ Saved {len(frame_metadata_list)} frames to storage")
        return overall_metadata

    def extract_frames(self, video, max_frames=0):
        """Main frame extraction function - returns metadata instead of frames"""
        
        print(f"\n=== Video Frame Extraction ===")
        print(f"Video object type: {type(video)}")
        print(f"Max frames: {max_frames if max_frames > 0 else 'All frames'}")
        
        # Handle different video input formats
        if isinstance(video, torch.Tensor):
            # Direct tensor format: [B, H, W, C] in range [0, 1]
            print(f"Processing video tensor with shape: {video.shape}")
            
            # Create metadata for tensor-based video
            metadata = self.create_tensor_metadata(video, max_frames)
            return (metadata,)
        
        elif isinstance(video, dict):
            # Handle different video dict formats
            frames_tensor = None
            
            if 'frames' in video:
                frames_tensor = video['frames']
            elif 'video' in video:
                frames_tensor = video['video']
            else:
                for key, value in video.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) == 4:
                        frames_tensor = value
                        break
            
            if frames_tensor is not None:
                print(f"Processing video dict with tensor shape: {frames_tensor.shape}")
                metadata = self.create_tensor_metadata(frames_tensor, max_frames)
                return (metadata,)
            else:
                raise ValueError(f"Could not find video frames in dict. Keys: {list(video.keys())}")
        
        else:
            # Try to extract video file path from video object
            print(f"Attempting to extract file path from video object...")
            print(f"Available methods: {[attr for attr in dir(video) if not attr.startswith('_')]}")
            
            video_path = self.get_video_file_path(video)
            
            if video_path:
                print(f"Found video file path: {video_path}")
                
                # Extract frames to storage and get metadata
                max_frames_to_extract = max_frames if max_frames > 0 else None
                metadata = self.extract_frames_to_storage(video_path, max_frames_to_extract)
                
                return (metadata,)
            
            else:
                raise ValueError(f"Could not extract video file path from video object. "
                               f"Video object type: {type(video)}. "
                               f"Available attributes: {[attr for attr in dir(video) if not attr.startswith('_')]}")

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VideoFrameExtractor": VideoFrameExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameExtractor": "Video Frame Extractor"
}