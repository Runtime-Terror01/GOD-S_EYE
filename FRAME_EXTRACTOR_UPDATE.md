# Frame Extractor Node Update

## Summary
Updated `FrameExtractorNode` to accept video input from ComfyUI's **Load Video** node instead of requiring a file path string.

## Changes Made

### 1. Input Type Change
**Before:**
```python
"video_path": ("STRING", {"default": "input/video.mp4"})
```

**After:**
```python
"video": ("VIDEO",)
```

### 2. Import VideoInput API
Added ComfyUI video input support:
```python
try:
    from comfy_api.input import VideoInput
    COMFYUI_AVAILABLE = True
except ImportError:
    VideoInput = None
    COMFYUI_AVAILABLE = False
```

### 3. Dual Input Support
The node now supports two input methods:

#### a) **ComfyUI VideoInput** (Primary)
- Accepts output from ComfyUI's "Load Video" node
- Directly processes video frames from VideoInput object
- More efficient - no file path issues

#### b) **String Path** (Legacy)
- Still supports file path strings for backwards compatibility
- Automatically strips quotes from paths
- Uses OpenCV to read video file

### 4. New Processing Methods

**`_process_video_input()`**: Handles VideoInput from ComfyUI
- Gets video properties: dimensions, FPS, frame count
- Iterates through video frames as tensors
- Converts RGB tensors to BGR numpy arrays for OpenCV
- Saves frames with metadata

**`_process_video_path()`**: Handles file path strings (original method)
- Validates file exists
- Uses OpenCV VideoCapture
- Extracts frames with stride

## Usage in ComfyUI

### Workflow Setup:
1. **Load Video** node → outputs VIDEO
2. **Frame Extractor** node → accepts VIDEO input
3. Connect Load Video output to Frame Extractor video input
4. Set output_dir and stride parameters
5. Run workflow

### Example Workflow:
```
[Load Video] → VIDEO → [Frame Extractor]
                           ├─ output_dir: "surveillance_storage"
                           ├─ stride: 2
                           ↓
                        FRAMES_BATCH → [Object Detection]
                        FRAME_META_BATCH → [Tracking]
```

## Benefits

✅ **No more file path errors** - Direct video input from ComfyUI  
✅ **Better integration** - Uses ComfyUI's native video type  
✅ **More flexible** - Works with any video source in ComfyUI  
✅ **Backwards compatible** - Still accepts string paths  
✅ **Proper quote handling** - Strips quotes from legacy path inputs

## Testing

To test the updated node:

1. Start ComfyUI: `python main.py`
2. Open ComfyUI in browser
3. Add "Load Video" node from `image/video` category
4. Add "Frame Extractor" node from `surveillance` category
5. Connect Load Video → Frame Extractor
6. Select a video file in Load Video node
7. Queue prompt and verify frames are extracted

## Error Handling

The node now properly handles:
- VideoInput objects from ComfyUI
- String paths with quotes (strips them)
- Missing ComfyUI API (graceful fallback)
- Invalid input types (clear error message)
