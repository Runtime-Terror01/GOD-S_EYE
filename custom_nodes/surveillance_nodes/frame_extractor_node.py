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

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.mpeg', '.mpg', '.m4v')

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
                    "default": 100,
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
    CATEGORY = "Surveillance"
    
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
        
        # Robustly get video path from various possible video input types
        video_path = self._get_video_path(video)
        
        # Validate file exists before opening
        if not os.path.exists(video_path):
            raise ValueError(f"Failed to open video: path does not exist: {video_path!r}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path!r} (cv2 couldn't open it)")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0
        
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
            frame_idx_int = int(frame_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_int)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Failed to read frame {frame_idx_int}")
                continue
            
            # Generate frame ID and path
            frame_id = str(uuid.uuid4())
            frame_filename = f"frame_{frame_idx_int:06d}.jpg"
            frame_path = session_dir / frame_filename
            
            # Save frame
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            
            # Calculate timestamp
            timestamp = frame_idx_int / fps if fps > 0 else 0.0
            
            # Create frame metadata
            frame_data = {
                'frame_id': frame_id,
                'frame_index': frame_idx_int,
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
        
        # Save metadata to JSON (exclude frames_in_memory for serialization)
        metadata_path = session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json_metadata = {k: v for k, v in frame_metadata.items() if k != 'frames_in_memory'}
            json.dump(json_metadata, f, indent=2)
        
        print(f"✓ Extracted {len(extracted_frames)} frames to {session_dir}")
        print(f"✓ Session ID: {session_id}")
        
        return (frame_metadata,)
    
    def _calculate_extraction_indices(self, mode: str, fps: float, total_frames: int,
                                     interval_seconds: float, max_frames: int) -> List[int]:
        """Calculate which frames to extract based on mode"""
        
        if total_frames <= 0:
            return [0]
        
        if mode == "all_frames":
            # Extract all frames up to max_frames (sample if too many)
            if total_frames <= max_frames:
                indices = list(range(0, total_frames))
            else:
                step = max(1, total_frames // max_frames)
                indices = list(range(0, total_frames, step))[:max_frames]
            
        elif mode == "interval":
            # Extract frames at regular time intervals
            if fps <= 0:
                # fallback to uniform sampling
                step = max(1, total_frames // max_frames)
                indices = list(range(0, total_frames, step))[:max_frames]
            else:
                interval_frames = max(1, int(round(fps * interval_seconds)))
                indices = list(range(0, total_frames, interval_frames))[:max_frames]
            
        elif mode == "keyframes":
            # Simple keyframe detection placeholder: uniform sampling
            step = max(1, total_frames // min(max_frames, max(1, total_frames // 10)))
            indices = list(range(0, total_frames, step))[:max_frames]
            
        else:
            indices = [0]  # Fallback to first frame
        
        return indices

    def _get_video_path(self, video_input) -> str:
        """
        Heuristic extraction of a filesystem path from various ComfyUI video input shapes.
        Will return a path only if os.path.exists(path) is True.
        """
        # 1) direct string or Path
        if isinstance(video_input, str):
            if os.path.exists(video_input):
                return video_input
        if isinstance(video_input, Path):
            if video_input.exists():
                return str(video_input)
        
        # 2) dict-like (has get)
        try:
            if hasattr(video_input, 'get') and callable(video_input.get):
                for key in ('video_path', 'path', 'file_path', 'filepath', 'filename', 'file'):
                    val = video_input.get(key)
                    if isinstance(val, (str, Path)) and os.path.exists(str(val)):
                        return str(val)
        except Exception:
            pass
        
        # 3) inspect __dict__/vars
        candidates = []
        try:
            v = vars(video_input)
            if isinstance(v, dict):
                for k, val in v.items():
                    if isinstance(val, (str, Path)) and os.path.exists(str(val)):
                        return str(val)
                    if isinstance(val, (str, Path)):
                        candidates.append((k, str(val)))
        except Exception:
            pass
        
        # 4) common attributes
        for attr in ('video_path', 'path', 'file_path', 'filepath', 'filename', 'file', 'stream', 'tempfile'):
            try:
                if hasattr(video_input, attr):
                    val = getattr(video_input, attr)
                    # handle file-like objects
                    if hasattr(val, 'name') and isinstance(val.name, str) and os.path.exists(val.name):
                        return val.name
                    if callable(val):
                        try:
                            val2 = val()
                            if isinstance(val2, (str, Path)) and os.path.exists(str(val2)):
                                return str(val2)
                        except Exception:
                            pass
                    if isinstance(val, (str, Path)) and os.path.exists(str(val)):
                        return str(val)
                    if isinstance(val, (str, Path)):
                        candidates.append((attr, str(val)))
            except Exception:
                pass
        
        # 5) scan dir() for any string-like attribute that looks like a video filename
        try:
            for name in dir(video_input):
                if name.startswith('__'):
                    continue
                try:
                    val = getattr(video_input, name)
                    # skip callables
                    if callable(val):
                        continue
                    if isinstance(val, (str, Path)):
                        s = str(val)
                        # if looks like a file name and exists - return
                        if s.lower().endswith(VIDEO_EXTENSIONS) and os.path.exists(s):
                            return s
                        # collect candidates for debug output
                        if s.lower().endswith(VIDEO_EXTENSIONS):
                            candidates.append((name, s))
                except Exception:
                    continue
        except Exception:
            pass
        
        # 6) last resort: try stringifying object and see if path exists inside
        try:
            s = str(video_input)
            # sometimes repr includes path; try to split and find substring ending with ext
            for ext in VIDEO_EXTENSIONS:
                if ext in s.lower():
                    # find full token containing ext
                    parts = s.replace('\\', '/').split('/')
                    for p in parts:
                        if p.lower().endswith(ext):
                            # try join prior parts to make a path guess
                            # try absolute and relative attempts
                            guess = p
                            if os.path.exists(guess):
                                return guess
            # otherwise, no luck here
        except Exception:
            pass
        
        # If we get here, we couldn't find an existing file path.
        # Provide helpful debugging info.
        debug_candidates = [f"{k}={v}" for k, v in candidates[:20]]
        raise ValueError(
            "Could not determine a valid existing video path from the provided video object.\n"
            f"Object type: {type(video_input)}\n"
            f"Sample candidate string-like attributes (up to 20): {debug_candidates}\n"
            "If you can, tell me which attribute of the VideoFromFile holds the path (or attach the node output repr)."
        )
        
# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VideoFrameExtractorNode": VideoFrameExtractorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameExtractorNode": "Extract Video Frames"
}
