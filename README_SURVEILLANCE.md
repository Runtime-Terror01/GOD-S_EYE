# Surveillance System - Node-Based Video Processing Platform

A complete node-based video surveillance system with ComfyUI-style backend nodes, FastAPI dashboard backend, and a minimal web frontend for monitoring and analysis.

## 🎯 Project Overview

This system extracts frames from video, runs YOLOv8 object detection, performs ByteTrack tracking, generates standardized metadata JSON files, uses LLM-based VLM analysis for comprehensive reports, and produces security alerts—all stored in a structured `surveillance_storage` layout for easy dashboard consumption.

## 📁 Project Structure

```
GOD'S_EYE/
├── nodes/                          # Core processing nodes
│   ├── __init__.py
│   ├── base_node.py               # Abstract base class
│   ├── data_types.py              # Standardized data types
│   ├── frame_extractor_node.py    # Video frame extraction
│   ├── object_detection_node.py   # YOLOv8 detection
│   ├── tracking_node.py           # ByteTrack tracking
│   ├── vlm_node.py                # LLM-based VLM analysis
│   └── alert_node.py              # Rule-based alert generation
├── pipelines/                      # Pipeline templates
│   ├── pipeline_template_basic.json
│   └── pipeline_template_fast.json
├── dashboard/                      # Dashboard components
│   ├── backend.py                 # FastAPI server
│   └── frontend.html              # Web UI
├── surveillance_storage/           # Data storage (auto-created)
│   └── session_<id>/
│       ├── raw_frame/
│       ├── annotated_frame/
│       ├── metadata/
│       └── vectors/
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install opencv-python numpy ultralytics fastapi uvicorn google-generativeai
```

### 2. Set Environment Variables

```bash
# Optional: For VLM analysis
set GEMINI_API_KEY=your_api_key_here

# Optional: Custom storage location
set SURVEILLANCE_ROOT=./surveillance_storage
```

### 3. Run Example Pipeline

```python
from nodes import (
    FrameExtractorNode,
    ObjectDetectionNode,
    TrackingNode,
    AlertNode,
    VLMNode
)

# Initialize nodes
frame_extractor = FrameExtractorNode({'session_id': 'demo'})
detector = ObjectDetectionNode({'conf_threshold': 0.5})
tracker = TrackingNode({'iou_threshold': 0.3})
alert_gen = AlertNode()
vlm = VLMNode({'session_id': 'demo'})

# Process video
video_path = "input/sample_video.mp4"
output_dir = "surveillance_storage"

# Extract frames
frames, frames_meta = frame_extractor.process(video_path, output_dir, stride=2)

# Detect objects
classes, bboxes, confs = detector.process(frames, frames_meta)

# Track objects
tracks, = tracker.process(classes, bboxes, confs, frames_meta)

# Generate alerts
alerts, = alert_gen.process(classes, bboxes, confs, tracks, frames_meta)

# Generate VLM report (optional)
report, = vlm.process(video_path)

# Save all metadata
session_id = frames_meta[0]['session_id']
session_path = f"{output_dir}/session_{session_id}"
detector.save_metadata(session_path)
tracker.save_metadata(session_path)
alert_gen.save_metadata(session_path)
vlm.save_metadata(session_path)

print(f"✅ Processing complete! Session: {session_id}")
print(f"📊 Frames: {len(frames)}")
print(f"🚨 Alerts: {len(alerts)}")
```

### 4. Start Dashboard

```bash
# Terminal 1: Start backend
python dashboard/backend.py

# Terminal 2: Serve frontend (or open frontend.html in browser)
# Navigate to http://localhost:8000 (backend provides CORS)
# Open dashboard/frontend.html in your browser
```

## 📊 Storage Layout

The system follows this exact storage structure:

```
surveillance_storage/
  session_<session_id>/
    raw_frame/
      frame_000000.jpg
      frame_000001.jpg
      ...
    annotated_frame/
      frame_000000.jpg
      ...
    metadata/
      session_metadata.json      # Video info, FPS, frame count
      frames_metadata.json        # Per-frame metadata array
      detection_metadata.json     # Detection results
      tracking_metadata.json      # Tracking results
      alert.json                  # Generated alerts
      report.json                 # VLM analysis report
    vectors/
      report_vectors.jsonl        # Vector DB ready format
```

## 🔄 Standardized Data Types

All nodes use these standardized Python types for interoperability:

- **VIDEO**: `str` (path) or `cv2.VideoCapture`
- **FRAMES_BATCH**: `List[np.ndarray]` or `np.ndarray(N,H,W,C)`
- **FRAME_META_BATCH**: `List[dict]` with `session_id`, `frame_id`, `frame_index`, `timestamp_sec`, `path`
- **CLASS_BATCH**: `List[List[str]]` - classes per frame
- **BBOX_BATCH**: `List[List[List[int]]]` - `[x1,y1,x2,y2]` per detection per frame
- **CONF_BATCH**: `List[List[float]]` - confidence per detection per frame
- **TRACKS_BATCH**: `List[List[dict]]` - tracks with `track_id`, `cls`, `bbox`, `conf`
- **REPORT**: `dict` with `session_id`, `comprehensive_report`, `summary_for_user`, `generated_at`
- **ALERTS**: `List[dict]` with `alert_id`, `timestamp_sec`, `frame_ref`, `reason`, `evidence`, `severity`

## 🔌 Dashboard API Endpoints

### Backend (FastAPI on port 8000)

```
GET  /sessions                            # List all sessions
GET  /sessions/{id}/frames                # Get frames metadata (paginated)
GET  /sessions/{id}/frames/{frame_id}    # Get frame image
GET  /sessions/{id}/detections            # Get detection data
GET  /sessions/{id}/tracks                # Get tracking data
GET  /sessions/{id}/alerts                # Get alerts
GET  /sessions/{id}/report                # Get VLM report
POST /search                              # Semantic search
GET  /health                              # Health check
```

### Frontend Features

- 📂 **Session List**: Browse all processing sessions
- 🎥 **Frame Viewer**: Navigate through extracted frames
- 🚨 **Alerts Panel**: Real-time alert monitoring (5s polling)
- 📊 **Session Overview**: Statistics and risk assessment
- 🔍 **Semantic Search**: Query reports and session data

## 🎬 Pipeline Templates

### Basic Pipeline (All Frames)
```json
// pipelines/pipeline_template_basic.json
// Processes every frame for maximum accuracy
```

### Fast Pipeline (Sampled Frames)
```json
// pipelines/pipeline_template_fast.json
// stride=5 for 5x faster processing
// Good for real-time monitoring or initial screening
```

## 🛠️ Node Configuration

### Frame Extractor
```python
config = {
    'session_id': 'custom_id'  # Optional, auto-generated if not provided
}
frame_extractor = FrameExtractorNode(config)
frames, meta = frame_extractor.process(video_path, output_dir, stride=1)
```

### Object Detection
```python
config = {
    'model_path': 'yolov8n.pt',  # Or custom model
    'conf_threshold': 0.5
}
detector = ObjectDetectionNode(config)
classes, bboxes, confs = detector.process(frames, frames_meta)
```

### Tracking
```python
config = {
    'iou_threshold': 0.3,
    'max_age': 30
}
tracker = TrackingNode(config)
tracks, = tracker.process(classes, bboxes, confs, frames_meta)
```

### Alert Generation
```python
config = {
    'rules': {
        'loitering': {'enabled': True, 'duration_threshold': 30.0},
        'suspicious_object': {'enabled': True, 'watchlist': ['knife', 'gun']},
        'crowding': {'enabled': True, 'person_threshold': 10}
    }
}
alert_gen = AlertNode(config)
alerts, = alert_gen.process(classes, bboxes, confs, tracks, frames_meta)
```

### VLM Analysis
```python
config = {
    'session_id': 'demo'
}
vlm = VLMNode(config)
report, = vlm.process(video_path, system_instruction, user_query)
```

## 📝 Metadata Formats

### session_metadata.json
```json
{
  "session_id": "abc123",
  "video_path": "input/video.mp4",
  "started_at": "2025-10-14T09:00:00+05:30",
  "frame_count": 1200,
  "fps": 30
}
```

### detection_metadata.json
```json
{
  "session_id": "abc123",
  "detections": {
    "frame_000000": [
      {"cls": "person", "bbox": [100,150,230,400], "conf": 0.92}
    ]
  }
}
```

### alert.json
```json
{
  "session_id": "abc123",
  "alerts": [
    {
      "alert_id": "a1",
      "timestamp_sec": 12.5,
      "frame_ref": "frame_000012",
      "reason": "loitering",
      "evidence": {"track_id": 5, "duration_sec": 31.2},
      "severity": "Medium"
    }
  ]
}
```

## 🔒 Security Alert Rules

1. **Loitering**: Person staying in small area > threshold seconds
2. **Suspicious Objects**: Weapons, knives, guns detected
3. **Crowding**: Person count exceeds threshold
4. **Restricted Zone**: (Configurable with zone definitions)

## 🧪 Testing

```python
# Run with mock data (no external dependencies)
from nodes import ObjectDetectionNode

# Mock detector will be used if ultralytics not available
detector = ObjectDetectionNode()
classes, bboxes, confs = detector.process(frames, frames_meta)
```

## 📦 Dependencies

- **opencv-python**: Video processing
- **numpy**: Array operations
- **ultralytics**: YOLOv8 detection (optional, has fallback)
- **fastapi**: Dashboard backend
- **uvicorn**: ASGI server
- **google-generativeai**: VLM analysis (optional, has mock)

## 🎯 Use Cases

1. **Retail Security**: Monitor for suspicious behavior and theft
2. **Traffic Monitoring**: Track vehicles and detect violations
3. **Facility Access**: Detect unauthorized access and loitering
4. **Event Security**: Crowd monitoring and threat detection
5. **Home Security**: Automated surveillance and alerts

## 🔧 Customization

### Add Custom Detection Classes
```python
# Modify watchlist in alert rules
config = {
    'rules': {
        'suspicious_object': {
            'watchlist': ['knife', 'gun', 'custom_class'],
            'conf_threshold': 0.6
        }
    }
}
```

### Custom Alert Rules
```python
# Extend AlertNode with custom logic
class CustomAlertNode(AlertNode):
    def _check_custom_rule(self, tracks, meta):
        # Your custom logic here
        pass
```

## 📈 Performance

- **Basic Pipeline**: Full accuracy, slower processing
- **Fast Pipeline**: 5x speedup with stride=5, may miss brief events
- **Real-time**: Adjust stride based on FPS requirements

## 🐛 Troubleshooting

**Q: YOLOv8 model not found?**
A: First run will auto-download yolov8n.pt. Or specify custom model path.

**Q: VLM analysis not working?**
A: Set `GEMINI_API_KEY` environment variable. Falls back to mock if unavailable.

**Q: Dashboard can't connect to backend?**
A: Ensure backend is running on port 8000 and CORS is enabled.

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

1. Follow the `BaseNode` interface for new nodes
2. Use standardized data types from `data_types.py`
3. Write metadata to proper storage locations
4. Add tests for new functionality

## 📞 Support

For issues and questions, please open a GitHub issue.

---

**Built with ❤️ for video surveillance and security applications**
