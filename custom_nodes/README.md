# ComfyUI Threat Detection Suite

RF-DETR based threat detection system for ComfyUI with video processing capabilities.

## Nodes

### 1. Load Threat Detection Model
- **Purpose**: Downloads and loads the RF-DETR threat detection model
- **Inputs**:
  - `model_name`: Model to load (default: "Subh775/Threat-Detection-RF-DETR")
  - `resolution`: Input resolution for the model (320-1280px, default: 640px)
- **Outputs**: `THREAT_MODEL` - Loaded model ready for inference

### 2. Video Frame Extractor
- **Purpose**: Extracts frames from video files for processing
- **Inputs**:
  - `video`: Video input (supports VideoFromFile objects and tensor formats)
  - `max_frames`: Maximum frames to extract (0 = all frames, useful for long videos)
- **Outputs**: `IMAGE` - Extracted frames as ComfyUI image tensor

### 3. Threat Detection
- **Purpose**: Detects threats in images using RF-DETR model and outputs JSON predictions
- **Inputs**:
  - `model`: Threat detection model from Load Model node
  - `image`: Images to process (from Frame Extractor or other image sources)
  - `threshold`: Detection confidence threshold (0.1-0.9, default: 0.5)
- **Outputs**: `STRING` - JSON formatted predictions with detection data

## Workflow

### For Video Processing:
1. **Video Input** → **Video Frame Extractor** → **Threat Detection** → **JSON Output**
2. Connect your video source to the Video Frame Extractor
3. Set `max_frames` if you want to limit processing time for long videos
4. Connect the extracted frames to the Threat Detection node
5. Load the model using Load Threat Detection Model node
6. Output will be JSON formatted predictions for each frame

### For Image Processing:
1. **Image Input** → **Threat Detection** → **JSON Output**
2. Connect your images directly to the Threat Detection node
3. Load the model using Load Threat Detection Model node
4. Output will be JSON formatted predictions for each image

## Detected Threat Classes

- **Gun** (Red)
- **Explosive** (Orange)
- **Grenade** (Red-Orange)
- **Knife** (Crimson)

## JSON Output Format

Each frame/image produces a prediction result in this format:

```json
{
  "inference_id": "0d40f9c5-c13c-4ee1-af6f-5025f3d098de",
  "time": 0.1744888450000417,
  "image": {
    "width": 320,
    "height": 240
  },
  "predictions": [
    {
      "x": 182.96666717529297,
      "y": 71.84080600738525,
      "width": 123.06419372558594,
      "height": 89.55509757995605,
      "confidence": 0.8885549902915955,
      "class": "gun",
      "class_id": 1,
      "detection_id": "f55ab2d6-277f-4841-a42d-74d0f0e73e41"
    }
  ]
}
```

## Performance Tips

- **Resolution**: Higher resolution (1280px) = better accuracy but slower processing
- **Max Frames**: For long videos, set max_frames to 100-500 to limit processing time
- **GPU**: Automatically uses GPU if available for faster processing

## Dependencies

```bash
pip install rfdetr==1.2.1 supervision==0.26.1
```

## Example Workflow

```
[Video Input] → [Video Frame Extractor] → [Threat Detection] → [JSON Output]
                      ↑                           ↑
                 max_frames=100              [Load Model]
                                                 ↑
                                            resolution=640
```

This setup allows you to process videos by first extracting frames, then running threat detection on those frames to get structured JSON predictions. The JSON output contains detailed information about each detection including bounding boxes, confidence scores, and unique IDs for tracking.