"""
Frame Extractor Node
Extracts frames from video and saves them with proper metadata
(robustified while preserving original inputs/outputs/metadata format)
"""

import os
import cv2
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone

from .base_node import BaseNode
from .data_types import VIDEO, FRAMES_BATCH, FRAME_META_BATCH

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.mpeg', '.mpg', '.m4v')

# Attempt to detect ComfyUI VideoInput type, but remain defensive
try:
    from comfy_api.input import VideoInput  # type: ignore
    COMFYUI_AVAILABLE = True
except Exception:
    VideoInput = None
    COMFYUI_AVAILABLE = False


class FrameExtractorNode(BaseNode):
    """
    Extract frames from video file and save to surveillance_storage layout
    Produces frames_metadata.json and session_metadata.json
    """

    # ComfyUI Node Configuration (kept same shape as original)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",),
                # "output_dir": ("STRING", {
                #     "default": "surveillance_storage",
                #     "multiline": False
                # }),
                "stride": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 30,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("FRAMES_BATCH", "FRAME_META_BATCH")
    RETURN_NAMES = ("frames", "frame_metadata")
    FUNCTION = "extract"
    CATEGORY = "surveillance"
    

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.session_id = None
        self.video_path = None
        self.fps = 0.0
        self.total_frames = 0
        self.output_dir = "surveillance_storage"

    def extract(self, video: Any, stride: int = 1):
        """ComfyUI entry point - accepts VideoInput from Load Video node or string path"""
        print(f"[FrameExtractor] Received inputs: video_type={type(video)}, output_dir={self.output_dir!r}, stride={stride}")
        return self.process(video, self.output_dir, stride)

    def process(self, video: Any, output_dir: str, stride: int = 1) -> Tuple[FRAMES_BATCH, FRAME_META_BATCH]:
        """
        Extract frames from video

        Args:
            video: VideoInput from ComfyUI's Load Video node, or path to video file
            output_dir: Base surveillance_storage directory
            stride: Extract every Nth frame (1 = all frames, 2 = every other frame, etc.)

        Returns:
            (frames_batch, frame_meta_batch):
                - frames_batch: List of numpy arrays (frames)
                - frame_meta_batch: List of frame metadata dictionaries
        """
        # Normalize and resolve output_dir to absolute Path to compute reliable relative paths later
        base_output_dir = Path(output_dir).resolve()
        base_output_dir.mkdir(parents=True, exist_ok=True)

        # Handle ComfyUI VideoInput or string path or other shapes via heuristics
        if COMFYUI_AVAILABLE and VideoInput is not None and isinstance(video, VideoInput):
            # If ComfyUI VideoInput subclass, try the dedicated handler (but that handler is defensive)
            return self._process_video_input(video, base_output_dir, stride)
        elif isinstance(video, (str, Path)):
            video_path = str(video).strip('"').strip("'")
            return self._process_video_path(video_path, base_output_dir, stride)
        else:
            # Try robust heuristics to extract a valid file path from `video`
            try:
                video_path = self._get_video_path(video)
                return self._process_video_path(video_path, base_output_dir, stride)
            except Exception as e:
                # If we couldn't get a path and object looks like a ComfyUI video,
                # try to process it as a ComfyUI video object anyway.
                # (This covers VideoFromFile types that are subclasses of VideoInput.)
                # Try _process_video_input as a last attempt.
                try:
                    return self._process_video_input(video, base_output_dir, stride)
                except Exception:
                    raise ValueError(f"Unsupported video input type: {type(video)}. Error: {e}")

    def _get_video_path(self, video_input) -> str:
        """
        Robust extraction of filesystem path from various video input shapes.
        Returns a path only if os.path.exists(path) is True.
        """
        # 1) direct string or Path
        if isinstance(video_input, str):
            candidate = video_input.strip('"').strip("'")
            if os.path.exists(candidate):
                return candidate
        if isinstance(video_input, Path):
            if video_input.exists():
                return str(video_input)

        # 2) dict-like (has get)
        candidates = []
        try:
            if hasattr(video_input, 'get') and callable(video_input.get):
                for key in ('video_path', 'path', 'file_path', 'filepath', 'filename', 'file', 'source'):
                    val = video_input.get(key)
                    if isinstance(val, (str, Path)) and os.path.exists(str(val)):
                        return str(val)
                    if isinstance(val, (str, Path)):
                        candidates.append((key, str(val)))
        except Exception:
            pass

        # 3) try vars() / __dict__
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

        # 4) common attribute names
        for attr in ('video_path', 'path', 'file_path', 'filepath', 'filename', 'file', 'source', 'stream', 'tempfile'):
            try:
                if hasattr(video_input, attr):
                    val = getattr(video_input, attr)
                    # file-like object with .name
                    if hasattr(val, 'name') and isinstance(val.name, str) and os.path.exists(val.name):
                        return val.name
                    # callable returning path
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
                    if callable(val):
                        continue
                    if isinstance(val, (str, Path)):
                        s = str(val)
                        if s.lower().endswith(VIDEO_EXTENSIONS) and os.path.exists(s):
                            return s
                        if s.lower().endswith(VIDEO_EXTENSIONS):
                            candidates.append((name, s))
                except Exception:
                    continue
        except Exception:
            pass

        # 6) last resort: inspect string form
        try:
            s = str(video_input)
            for ext in VIDEO_EXTENSIONS:
                if ext in s.lower():
                    parts = s.replace('\\', '/').split('/')
                    for p in parts:
                        if p.lower().endswith(ext) and os.path.exists(p):
                            return p
        except Exception:
            pass

        debug_candidates = [f"{k}={v}" for k, v in candidates[:20]]
        raise ValueError(
            "Could not determine a valid existing video path from the provided video object.\n"
            f"Object type: {type(video_input)}\n"
            f"Sample candidate string-like attributes (up to 20): {debug_candidates}\n"
            "If possible, pass a plain file path (str) or the ComfyUI Load Video node output."
        )

    def _process_video_input(self, video: Any, base_output_dir: Path, stride: int) -> Tuple[FRAMES_BATCH, FRAME_META_BATCH]:
        """
        Process a ComfyUI VideoInput (defensive).
        If the ComfyUI object doesn't expose get_images() or reports fps/frame_count == 0,
        attempt to fall back to a filesystem path (via heuristics) and use OpenCV instead.
        """
        # Generate session ID
        self.session_id = self.config.get('session_id') or self._generate_session_id()

        # Create session directory structure
        storage_path = base_output_dir / f"session_{self.session_id}"
        raw_frame_dir = storage_path / "raw_frame"
        raw_frame_dir.mkdir(parents=True, exist_ok=True)

        # Defensive retrieval of properties
        try:
            width, height = video.get_dimensions()
        except Exception:
            width, height = None, None

        try:
            fps = float(video.get_fps())
        except Exception:
            fps = 0.0

        try:
            total_frames = int(video.get_num_frames())
        except Exception:
            total_frames = 0

        self.fps = fps
        self.total_frames = total_frames
        self.video_path = getattr(video, 'source', getattr(video, 'file', f"ComfyUI_Video_{self.session_id}"))

        print(f"[FrameExtractor] Extracting frames from ComfyUI video (stride={stride}). Total frames (reported): {total_frames}, fps: {fps}")

        # If object does not provide get_images(), or fps/total_frames are 0, try fallback to actual file path
        images_iter = getattr(video, 'get_images', None)
        if not callable(images_iter) or (total_frames == 0 and fps == 0.0):
            # Try to obtain a real file path and fallback to OpenCV path-based extractor
            try:
                guessed_path = self._get_video_path(video)
                print(f"[FrameExtractor] Falling back to file path extracted from object: {guessed_path}")
                return self._process_video_path(guessed_path, base_output_dir, stride)
            except Exception as e:
                # No path found - raise clearer error with diagnostics
                raise ValueError("Provided ComfyUI video object does not expose get_images(); cannot extract frames. "
                                 "Attempted fallback to detect file path but failed. " + str(e))

        # Otherwise use get_images() generator
        frames_batch: List[np.ndarray] = []
        frame_meta_batch: List[Dict[str, Any]] = []
        extracted_count = 0
        frame_index = 0

        for frame_tensor in video.get_images():
            if frame_index % stride == 0:
                # Convert tensor/array to numpy uint8 BGR
                if hasattr(frame_tensor, 'cpu') and hasattr(frame_tensor, 'numpy'):
                    frame_np = (frame_tensor.cpu().numpy() * 255).astype(np.uint8)
                else:
                    frame_np = np.asarray(frame_tensor)
                    if frame_np.dtype != np.uint8:
                        try:
                            frame_np = (frame_np * 255).astype(np.uint8)
                        except Exception:
                            frame_np = frame_np.astype(np.uint8)

                # Ensure channels correct: if RGB -> convert to BGR
                if frame_np.ndim == 3 and frame_np.shape[2] == 3:
                    try:
                        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    except Exception:
                        frame_bgr = frame_np
                elif frame_np.ndim == 2:
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame_np

                frame_id = str(uuid.uuid4())
                frame_filename = f"frame_{frame_index:06d}.jpg"
                frame_path = raw_frame_dir / frame_filename
                cv2.imwrite(str(frame_path), frame_bgr)

                timestamp_sec = frame_index / fps if fps > 0 else 0.0

                # Relative path with respect to base_output_dir so dashboard can construct absolute path
                try:
                    rel = frame_path.resolve().relative_to(base_output_dir.resolve())
                    rel_path_str = rel.as_posix()
                except Exception:
                    rel_path_str = str(frame_path.resolve())

                frame_meta = {
                    "session_id": self.session_id,
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "path": rel_path_str
                }

                frames_batch.append(frame_bgr)
                frame_meta_batch.append(frame_meta)
                extracted_count += 1

            frame_index += 1

        print(f"[FrameExtractor] Extracted {extracted_count} frames (every {stride} frame(s)).")

        # Save metadata in same format expected by dashboard
        self.metadata = {
            "session_metadata": {
                "session_id": self.session_id,
                "video_path": self.video_path,
                "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "frame_count": extracted_count,
                "fps": fps,
                "resolution": f"{width}x{height}" if width and height else None
            },
            "frames_metadata": frame_meta_batch
        }

        self.save_metadata(str(storage_path))

        return frames_batch, frame_meta_batch

    def _process_video_path(self, video_path: str, base_output_dir: Path, stride: int) -> Tuple[FRAMES_BATCH, FRAME_META_BATCH]:
        """
        Extract frames from video file path
        """
        self.video_path = video_path

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Generate session ID
        self.session_id = self.config.get('session_id') or self._generate_session_id()

        # Create directories
        storage_path = base_output_dir / f"session_{self.session_id}"
        raw_frame_dir = storage_path / "raw_frame"
        raw_frame_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frames_batch: List[np.ndarray] = []
        frame_meta_batch: List[Dict[str, Any]] = []
        frame_index = 0
        extracted_count = 0

        print(f"[FrameExtractor] Extracting frames from {video_path} (stride={stride}). Total frames: {self.total_frames}, fps: {self.fps}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % stride == 0:
                frame_id = str(uuid.uuid4())
                frame_filename = f"frame_{frame_index:06d}.jpg"
                frame_path = raw_frame_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)

                timestamp_sec = frame_index / self.fps if self.fps > 0 else 0.0

                try:
                    rel = frame_path.resolve().relative_to(base_output_dir.resolve())
                    rel_path_str = rel.as_posix()
                except Exception:
                    rel_path_str = str(frame_path.resolve())

                frame_meta = {
                    "session_id": self.session_id,
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "path": rel_path_str
                }

                frames_batch.append(frame)
                frame_meta_batch.append(frame_meta)
                extracted_count += 1

            frame_index += 1

        cap.release()

        print(f"[FrameExtractor] Extracted {extracted_count} frames (every {stride} frame(s)).")

        self.metadata = {
            "session_metadata": {
                "session_id": self.session_id,
                "video_path": video_path,
                "started_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "frame_count": extracted_count,
                "fps": self.fps,
                "resolution": f"{width}x{height}"
            },
            "frames_metadata": frame_meta_batch
        }

        self.save_metadata(str(storage_path))

        return frames_batch, frame_meta_batch

    def _generate_session_id(self) -> str:
        """Generate a short session ID"""
        return str(uuid.uuid4())[:8]

    def _write_metadata(self, metadata_dir: Path) -> None:
        """Write session_metadata.json and frames_metadata.json"""
        session_meta_path = metadata_dir / "session_metadata.json"
        self._save_json(session_meta_path, self.metadata['session_metadata'])

        frames_meta_path = metadata_dir / "frames_metadata.json"
        self._save_json(frames_meta_path, self.metadata['frames_metadata'])

        print(f"[FrameExtractor] Saved metadata to {metadata_dir}")
