# 🚀 Quick Start Guide

## Installation (5 minutes)

### 1. Install Python Dependencies

```bash
pip install -r requirements_surveillance.txt
```

### 2. Set API Key (Optional - for VLM)

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

**Windows CMD:**
```cmd
set GEMINI_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY=your_api_key_here
```

## Running Your First Pipeline (2 minutes)

### Method 1: Using Example Script

```bash
# Basic usage (fastest)
python example_pipeline.py input/your_video.mp4

# With VLM analysis
python example_pipeline.py input/your_video.mp4 --vlm

# Custom stride (1=all frames, 5=every 5th frame)
python example_pipeline.py input/your_video.mp4 --stride 5
```

### Method 2: Python Code

Create `test_pipeline.py`:

```python
from nodes import (
    FrameExtractorNode,
    ObjectDetectionNode,
    TrackingNode,
    AlertNode
)

# Setup
video_path = "input/sample.mp4"
output_dir = "surveillance_storage"

# Extract frames (every 2nd frame)
extractor = FrameExtractorNode()
frames, meta = extractor.process(video_path, output_dir, stride=2)

# Detect objects
detector = ObjectDetectionNode({'conf_threshold': 0.5})
classes, bboxes, confs = detector.process(frames, meta)

# Track objects
tracker = TrackingNode()
tracks, = tracker.process(classes, bboxes, confs, meta)

# Generate alerts
alert_gen = AlertNode()
alerts, = alert_gen.process(classes, bboxes, confs, tracks, meta)

# Save metadata
session_id = meta[0]['session_id']
session_path = f"{output_dir}/session_{session_id}"
detector.save_metadata(session_path)
tracker.save_metadata(session_path)
alert_gen.save_metadata(session_path)

print(f"✅ Done! Session: {session_id}")
print(f"📊 Frames: {len(frames)}, Alerts: {len(alerts)}")
```

Run it:
```bash
python test_pipeline.py
```

## Viewing Results (1 minute)

### Start Dashboard

**Terminal 1 - Backend:**
```bash
python dashboard/backend.py
```

**Terminal 2 - Open Frontend:**
- Open `dashboard/frontend.html` in your browser
- Or navigate to: `http://localhost:8000` (API docs)

### What You'll See

1. **Session List** (left panel): All processed videos
2. **Frame Viewer** (center): Browse extracted frames
3. **Alerts Panel** (right): Security alerts
4. **Search Box** (bottom): Query VLM reports

## Test with Sample Data

### No video? Use webcam!

```python
import cv2

# Capture 10 seconds from webcam
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('test_video.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      30, (640, 480))

for _ in range(300):  # 10 seconds at 30fps
    ret, frame = cap.read()
    if ret:
        out.write(frame)

cap.release()
out.release()

# Now process it
from example_pipeline import run_surveillance_pipeline
run_surveillance_pipeline('test_video.mp4')
```

## Common Issues

### ❌ "Video file not found"
**Solution:** Check video path. Use absolute path if needed.
```python
import os
video_path = os.path.abspath('input/video.mp4')
```

### ❌ "ultralytics not available"
**Solution:** System uses mock detector. Install for real detection:
```bash
pip install ultralytics
```

### ❌ "GEMINI_API_KEY not set"
**Solution:** VLM uses mock. Get free API key from:
https://makersuite.google.com/app/apikey

### ❌ Dashboard shows "No sessions"
**Solution:** Run pipeline first to generate data:
```bash
python example_pipeline.py input/video.mp4
```

### ❌ CORS error in browser
**Solution:** Backend automatically enables CORS. Make sure it's running on port 8000.

## Performance Tips

### 🚀 Fast Processing
```bash
# Use larger stride for faster processing
python example_pipeline.py video.mp4 --stride 5
```

### 🎯 Accurate Processing
```bash
# Use stride=1 for maximum accuracy
python example_pipeline.py video.mp4 --stride 1
```

### 💾 Large Videos
```python
# Process in chunks
extractor = FrameExtractorNode()
# Will automatically handle large videos
```

## Next Steps

1. ✅ Run example pipeline
2. ✅ View results in dashboard
3. 📝 Customize alert rules
4. 🔧 Add custom detection classes
5. 🚀 Deploy to production

## Directory Structure After First Run

```
surveillance_storage/
  session_abc123/
    raw_frame/          # ← Extracted frames
    annotated_frame/    # ← Frames with detection boxes
    metadata/
      session_metadata.json
      frames_metadata.json
      detection_metadata.json
      tracking_metadata.json
      alert.json
      report.json       # ← If VLM enabled
    vectors/
      report_vectors.jsonl
```

## API Quick Reference

### Process Video
```python
from nodes import FrameExtractorNode
node = FrameExtractorNode()
frames, meta = node.process(video_path, output_dir, stride=2)
```

### Detect Objects
```python
from nodes import ObjectDetectionNode
node = ObjectDetectionNode({'conf_threshold': 0.5})
classes, bboxes, confs = node.process(frames, meta)
```

### Generate Alerts
```python
from nodes import AlertNode
node = AlertNode()
alerts, = node.process(classes, bboxes, confs, tracks, meta)
```

### Query Dashboard
```bash
# List sessions
curl http://localhost:8000/sessions

# Get alerts
curl http://localhost:8000/sessions/abc123/alerts

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "suspicious", "limit": 10}'
```

## Support

Need help? Check:
1. README_SURVEILLANCE.md (detailed docs)
2. Example code in `example_pipeline.py`
3. API docs at `http://localhost:8000/docs`

---

**Ready to start? Run this:**

```bash
python example_pipeline.py input/your_video.mp4
python dashboard/backend.py
# Open dashboard/frontend.html
```

🎉 **You're all set!**
