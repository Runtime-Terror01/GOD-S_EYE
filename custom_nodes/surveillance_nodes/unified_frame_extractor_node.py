"""
UnifiedVideoFrameExtractorNode
- Combines robustness (temp-file support, logging, JSON-safe metadata) with
  tensor/dict support and efficient sequential read.
- Default: sequential read (fast for many codecs). Set random_access=True if you
  need specific frame indices.
"""

import os
import re
import uuid
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import cv2
from PIL import Image
import numpy as np
import torch

VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.mpeg', '.mpg', '.m4v')

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class UnifiedVideoFrameExtractor:
    def __init__(self, storage_root: str = None):
        self.storage_root = Path(storage_root) if storage_root else Path.cwd() / "storage"
        self.storage_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000}),
            },
            "optional": {
                "quality": ("INT", {"default": 90, "min": 50, "max": 100}),
                "keep_frames_in_memory": ("BOOL", {"default": False}),
                "random_access": ("BOOL", {"default": False, "tooltip": "Use cap.set to seek frames (may be slow)."}),
            }
        }

    RETURN_TYPES = ("FRAME_METADATA",)
    RETURN_NAMES = ("frame_metadata",)
    FUNCTION = "extract_frames"
    CATEGORY = "Video Processing"

    # -------------------------
    # Public API
    # -------------------------
    def extract_frames(self, video, max_frames: int = 0, quality: int = 90,
                       keep_frames_in_memory: bool = False, random_access: bool = False) -> Tuple[Dict]:
        """Main entry used by ComfyUI. Returns (metadata_dict,)"""

        # Handle tensor input or dict-of-tensors quickly
        if isinstance(video, torch.Tensor):
            logger.info("Detected torch.Tensor input with shape %s", tuple(video.shape))
            return (self._create_tensor_metadata(video, max_frames, quality),)

        if isinstance(video, dict):
            # Try to find a tensor inside the dict
            tensor = None
            for k, v in video.items():
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    tensor = v
                    break
            if tensor is not None:
                logger.info("Detected tensor inside dict under key '%s'", k)
                return (self._create_tensor_metadata(tensor, max_frames, quality),)

        # Otherwise attempt to find a filesystem path
        temp_file = None
        try:
            video_path = self._get_video_path(video)
        except ValueError as e:
            # If object has a read() (file-like), write to temp file
            if hasattr(video, "read"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_file = tmp.name
                logger.info("Writing file-like video to temp file: %s", temp_file)
                try:
                    # Some file-like objects return bytes in one read; others need iterative reads
                    try:
                        data = video.read()
                        if isinstance(data, (bytes, bytearray)):
                            tmp.write(data)
                        else:
                            video.seek(0)
                            while True:
                                chunk = video.read(65536)
                                if not chunk:
                                    break
                                tmp.write(chunk)
                    except Exception:
                        video.seek(0)
                        while True:
                            chunk = video.read(65536)
                            if not chunk:
                                break
                            tmp.write(chunk)
                finally:
                    tmp.close()
                video_path = temp_file
            else:
                # Re-raise the diagnostic ValueError with candidate info
                raise

        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"Video path not found or accessible: {video_path!r}")

        metadata = self._extract_from_file(video_path, max_frames, quality,
                                           keep_frames_in_memory, random_access)

        # cleanup temp
        if temp_file:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

        return (metadata,)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _extract_from_file(self, video_path: str, max_frames: int, quality: int,
                           keep_frames_in_memory: bool, random_access: bool) -> Dict:

        video_path = os.path.abspath(video_path)
        # make unique session dir
        session_id = str(uuid.uuid4())[:8]
        video_name = Path(video_path).stem
        session_dir = self.storage_root / f"{video_name}_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"cv2 failed to open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0

        logger.info("Video opened: %s — %d frames @ %.2f FPS (%dx%d)", video_path, total_frames, fps, width, height)

        # If max_frames == 0 => all frames; else sample to max_frames by step
        if max_frames and max_frames > 0 and total_frames > 0 and max_frames < total_frames:
            step = max(1, total_frames // max_frames)
            logger.info("Sampling mode: step=%d to limit to ~%d frames", step, max_frames)
        else:
            step = 1

        frame_metadata_list = []
        frames_in_memory = [] if keep_frames_in_memory else None
        frame_idx = 0
        extracted = 0
        progress_print_step = max(1, total_frames // 10) if total_frames > 0 else 100

        try:
            if random_access:
                # Seek to each frame index using step
                idx = 0
                while True:
                    target = idx * step
                    if total_frames > 0 and target >= total_frames:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
                    ret, frame = cap.read()
                    if not ret:
                        logger.debug("Failed to read at index %d", target)
                        # safety: advance idx and continue
                        idx += 1
                        if total_frames > 0 and idx * step >= total_frames:
                            break
                        continue

                    # save
                    meta = self._save_frame(frame, session_dir, extracted, target, fps, quality)
                    frame_metadata_list.append(meta)
                    if keep_frames_in_memory:
                        frames_in_memory.append({"frame": frame.copy(), "metadata": meta})
                    extracted += 1
                    idx += 1
                    if max_frames and extracted >= max_frames:
                        break

            else:
                # Sequential read: fast for streaming and most codecs
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % step == 0:
                        meta = self._save_frame(frame, session_dir, extracted, frame_idx, fps, quality)
                        frame_metadata_list.append(meta)
                        if keep_frames_in_memory:
                            frames_in_memory.append({"frame": frame.copy(), "metadata": meta})
                        extracted += 1
                        if max_frames and extracted >= max_frames:
                            break
                    frame_idx += 1
                    # progress prints
                    if (total_frames > 0 and frame_idx % progress_print_step == 0) or (frame_idx % 500 == 0 and total_frames == 0):
                        logger.info("Progress: read %d frames, extracted %d", frame_idx, extracted)
        finally:
            cap.release()

        # Build JSON-serializable metadata
        meta_out = {
            "session_id": session_id,
            "video_source": video_path,
            "storage_directory": str(session_dir),
            "extraction_timestamp": datetime.now().isoformat(),
            "video_properties": {"fps": float(fps), "total_frames": int(total_frames), "duration": float(duration), "resolution": {"width": width, "height": height}},
            "extraction_settings": {"random_access": bool(random_access), "step": int(step), "quality": int(quality)},
            "frames": frame_metadata_list,
            "total_frames_extracted": len(frame_metadata_list)
        }

        # Write metadata.json
        metadata_path = session_dir / "metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(meta_out, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write metadata.json: %s", e)

        # If requested, attach a light summary of in-memory frames (not full pixel arrays)
        if keep_frames_in_memory and frames_in_memory is not None:
            meta_out["frames_in_memory"] = [{"metadata": f["metadata"], "shape": getattr(f["frame"], "shape", None)} for f in frames_in_memory]
            meta_out["_frames_blob"] = frames_in_memory  # note: this is not JSON-serializable

        return meta_out

    def _save_frame(self, frame: np.ndarray, session_dir: Path, extracted_count: int, original_frame_number: int, fps: float, quality: int) -> Dict:
        # unique id and filename
        frame_id = str(uuid.uuid4())
        filename = f"frame_{original_frame_number:06d}_{frame_id[:8]}.jpg"
        path = session_dir / filename

        # Convert BGR -> RGB for PIL (cv2.imwrite could also be used directly)
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.save(path, "JPEG", quality=int(quality))
        except Exception:
            # fallback: attempt cv2.imwrite (BGR)
            try:
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            except Exception as e:
                logger.error("Failed to save frame %s: %s", path, e)
                raise

        timestamp = (original_frame_number / fps) if fps > 0 else 0.0
        meta = {
            "frame_id": frame_id,
            "frame_index": extracted_count,
            "original_frame_number": int(original_frame_number),
            "timestamp": float(timestamp),
            "path": str(path),
            "width": int(frame.shape[1]) if hasattr(frame, "shape") else None,
            "height": int(frame.shape[0]) if hasattr(frame, "shape") else None,
            "session_id": None  # filled by caller if needed
        }
        return meta

    def _create_tensor_metadata(self, tensor: torch.Tensor, max_frames: int, quality: int) -> Dict:
        # Accept tensor as [B, H, W, C] or [B, C, H, W]
        t = tensor.detach().cpu()
        if t.dim() == 4 and t.shape[1] in (1, 3):  # assume [B, C, H, W]
            t = t.permute(0, 2, 3, 1)  # to [B,H,W,C]
        # normalize to 0-255
        arr = (t.numpy() * 255).astype(np.uint8) if t.dtype != np.uint8 else t.numpy()

        total = arr.shape[0]
        if max_frames and max_frames > 0 and max_frames < total:
            idxs = np.linspace(0, total - 1, num=max_frames).astype(int)
            arr = arr[idxs]
            logger.info("Sampled %d frames from tensor input", len(arr))

        session_id = str(uuid.uuid4())[:8]
        video_name = f"tensor_{session_id}"
        session_dir = self.storage_root / video_name
        session_dir.mkdir(parents=True, exist_ok=True)

        frames_meta = []
        for i, frame_np in enumerate(arr):
            frame_id = str(uuid.uuid4())
            fn = f"frame_{i:06d}_{frame_id[:8]}.jpg"
            path = session_dir / fn
            Image.fromarray(frame_np).save(path, "JPEG", quality=int(quality))
            frames_meta.append({
                "frame_id": frame_id,
                "frame_index": i,
                "original_frame_number": i,
                "timestamp": i * (1.0 / 30.0),
                "path": str(path),
                "width": int(frame_np.shape[1]),
                "height": int(frame_np.shape[0]),
                "video_source": "tensor_input",
                "session_id": session_id
            })

        out = {
            "session_id": session_id,
            "video_source": "tensor_input",
            "storage_directory": str(session_dir),
            "extraction_timestamp": datetime.now().isoformat(),
            "fps": 30.0,
            "duration": len(frames_meta) / 30.0,
            "frames": frames_meta,
            "total_frames_extracted": len(frames_meta)
        }
        return out

    # -------------------------
    # Path discovery (combined heuristics)
    # -------------------------
    def _get_video_path(self, obj) -> str:
        # 1) direct str / Path
        if isinstance(obj, str):
            if os.path.exists(obj):
                return obj
        if isinstance(obj, Path):
            if obj.exists():
                return str(obj)

        # 2) dict-like
        try:
            if hasattr(obj, "get") and callable(obj.get):
                for key in ('video_path', 'path', 'file_path', 'filepath', 'filename', 'file', 'source'):
                    val = obj.get(key)
                    if isinstance(val, (str, Path)) and os.path.exists(str(val)):
                        return str(val)
        except Exception:
            pass

        # 3) attributes / methods
        file_attrs = ['file', 'path', 'filename', '_file', 'video_path', 'filepath', 'source']
        for attr in file_attrs:
            try:
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    # file-like objects with .name
                    if hasattr(val, "name") and isinstance(val.name, str) and os.path.exists(val.name):
                        return val.name
                    if callable(val):
                        try:
                            v2 = val()
                            if isinstance(v2, (str, Path)) and os.path.exists(str(v2)):
                                return str(v2)
                        except Exception:
                            pass
                    if isinstance(val, (str, Path)) and os.path.exists(str(val)):
                        return str(val)
            except Exception:
                continue

        # 4) ComfyUI-specific heuristics: get_stream_source / get_components / save_to
        try:
            if hasattr(obj, "get_stream_source"):
                try:
                    s = obj.get_stream_source()
                    if isinstance(s, str) and os.path.exists(s):
                        return s
                except Exception:
                    pass
            if hasattr(obj, "get_components"):
                try:
                    comps = obj.get_components()
                    if isinstance(comps, dict):
                        for k in ('path', 'file', 'source', 'filename'):
                            if k in comps and os.path.exists(comps[k]):
                                return comps[k]
                    elif isinstance(comps, (list, tuple)) and len(comps) > 0:
                        if isinstance(comps[0], str) and os.path.exists(comps[0]):
                            return comps[0]
                except Exception:
                    pass
            if hasattr(obj, "save_to"):
                try:
                    tmp = "temp_video_check.mp4"
                    obj.save_to(tmp)
                    if os.path.exists(tmp):
                        os.remove(tmp)
                        return tmp
                except Exception:
                    pass
        except Exception:
            pass

        # 5) parse __str__ for paths
        s = str(obj)
        for ext in VIDEO_EXTS:
            if ext in s.lower():
                # crude split tokens
                toks = re.split(r'[,\s;\"\']+', s)
                for t in toks:
                    if t.lower().endswith(ext) and os.path.exists(t):
                        return t

        # failed
        raise ValueError("Could not determine a valid existing video path from the provided video object. "
                         f"Object type: {type(obj)}. Try passing a file path or a tensor/dict.")

# Node mapping (for ComfyUI)
NODE_CLASS_MAPPINGS = {
    "UnifiedVideoFrameExtractor": UnifiedVideoFrameExtractor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedVideoFrameExtractor": "Unified Video Frame Extractor"
}
