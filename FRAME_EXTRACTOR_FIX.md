# Frame Extractor Node - Parameter Validation Fix

## Issue
```
Failed to convert an input value to a INT value: stride, surveillance_storage, 
invalid literal for int() with base 10: 'surveillance_storage'
```

ComfyUI was trying to parse the `output_dir` string value as the `stride` INT parameter, suggesting parameter misalignment.

## Root Cause
The INPUT_TYPES dictionary format may not have been explicit enough for ComfyUI's parameter parser, causing parameters to be passed in the wrong order.

## Fixes Applied

### 1. Explicit Parameter Definitions
Updated INPUT_TYPES to use more explicit parameter configuration:

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "video": ("VIDEO",),
            "output_dir": ("STRING", {
                "default": "surveillance_storage",
                "multiline": False
            }),
            "stride": ("INT", {
                "default": 2,
                "min": 1,
                "max": 30,
                "step": 1
            }),
        }
    }
```

### 2. Robust Video Path Extraction
Added `_get_video_path()` method to handle various video input formats:

- Direct string/Path objects
- Dict-like objects with video path keys
- Objects with video path attributes
- Scans for attributes ending in video extensions

### 3. Enhanced Error Handling
Added fallback logic in `process()` method:

```python
if COMFYUI_AVAILABLE and isinstance(video, VideoInput):
    return self._process_video_input(video, output_dir, stride)
elif isinstance(video, str):
    video_path = video.strip('"').strip("'")
    return self._process_video_path(video_path, output_dir, stride)
else:
    # Try robust path extraction
    video_path = self._get_video_path(video)
    return self._process_video_path(video_path, output_dir, stride)
```

### 4. Debug Logging
Added debug output to track parameter values:

```python
def extract(self, video: Any, output_dir: str, stride: int = 1):
    print(f"[FrameExtractor] Received inputs:")
    print(f"  - video type: {type(video)}")
    print(f"  - output_dir: {output_dir!r}")
    print(f"  - stride: {stride}")
    return self.process(video, output_dir, stride)
```

## Testing

1. Restart ComfyUI: `python main.py`
2. Check console output for any import errors
3. Add nodes to workflow:
   - Load Video node
   - Frame Extractor node
4. Connect Load Video → Frame Extractor
5. Check console for debug output showing correct parameter values
6. Queue prompt

## Expected Console Output

```
[FrameExtractor] Received inputs:
  - video type: <class 'comfy_api.input_impl.VideoFromFile'>
  - output_dir: 'surveillance_storage'
  - stride: 2
Extracting frames from ComfyUI video (stride=2)
Total frames: 1234, FPS: 30.0, Resolution: 1920x1080
...
```

## If Error Persists

Check:
1. Are parameters in the correct order in ComfyUI UI?
2. Is the video input connected to the VIDEO socket (not STRING)?
3. Check browser console for JavaScript errors
4. Try clearing ComfyUI cache and refreshing browser
5. Check terminal output for the debug print statements

## Additional Notes

- The node now supports multiple video input formats
- Backward compatible with string paths
- Automatically strips quotes from paths
- More informative error messages
