# System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SURVEILLANCE SYSTEM                              │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: video.mp4
   │
   ▼
┌──────────────────────────┐
│  FrameExtractorNode      │  ← Extracts frames (configurable stride)
│  • stride=1 (all frames) │
│  • stride=5 (fast mode)  │
└──────────────────────────┘
   │
   │ FRAMES_BATCH + FRAME_META_BATCH
   │ (List[np.ndarray], List[dict])
   ▼
┌──────────────────────────┐
│  ObjectDetectionNode     │  ← YOLOv8 detection
│  • Detect objects        │
│  • Generate bboxes       │
│  • Confidence scores     │
└──────────────────────────┘
   │
   │ CLASS_BATCH + BBOX_BATCH + CONF_BATCH
   │ (List[List[str]], List[List[List[int]]], List[List[float]])
   ▼
┌──────────────────────────┐
│  TrackingNode            │  ← ByteTrack / IoU tracker
│  • Maintain track IDs    │
│  • Track trajectories    │
│  • Handle occlusions     │
└──────────────────────────┘
   │
   │ TRACKS_BATCH
   │ (List[List[dict]])
   ▼
┌──────────────────────────┐
│  AlertNode               │  ← Rule-based alerts
│  • Loitering detection   │
│  • Weapon detection      │
│  • Crowding detection    │
└──────────────────────────┘
   │
   │ ALERTS
   │ (List[dict])
   ▼

PARALLEL PROCESSING:
┌──────────────────────────┐
│  VLMNode (Optional)      │  ← LLM analysis
│  • Comprehensive report  │
│  • Risk assessment       │
│  • Recommendations       │
└──────────────────────────┘
   │
   │ REPORT
   │ (dict)
   ▼

┌─────────────────────────────────────────────────────────────────────────┐
│                    SURVEILLANCE STORAGE STRUCTURE                       │
├─────────────────────────────────────────────────────────────────────────┤
│  surveillance_storage/                                                  │
│    session_<session_id>/                                                │
│      ├── raw_frame/              ← Original extracted frames            │
│      │     ├── frame_000000.jpg                                         │
│      │     ├── frame_000001.jpg                                         │
│      │     └── ...                                                      │
│      ├── annotated_frame/        ← Frames with detection boxes          │
│      │     ├── frame_000000.jpg                                         │
│      │     └── ...                                                      │
│      ├── metadata/               ← All JSON metadata                    │
│      │     ├── session_metadata.json     (video info, fps, count)      │
│      │     ├── frames_metadata.json      (per-frame metadata)          │
│      │     ├── detection_metadata.json   (detection results)           │
│      │     ├── tracking_metadata.json    (tracking results)            │
│      │     ├── alert.json                (generated alerts)            │
│      │     └── report.json               (VLM analysis)                │
│      └── vectors/                ← Vector DB ready format               │
│            └── report_vectors.jsonl                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         DASHBOARD ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│   Frontend (HTML/JS)     │  ← Single page application
│   dashboard/frontend.html│
│   • Session list         │
│   • Frame viewer         │
│   • Alerts panel         │
│   • Search interface     │
└──────────────────────────┘
        │
        │ HTTP Requests (JSON)
        ▼
┌──────────────────────────┐
│   Backend (FastAPI)      │  ← REST API server
│   dashboard/backend.py   │
│   Port: 8000             │
│   • CORS enabled         │
└──────────────────────────┘
        │
        │ File System Access
        ▼
┌──────────────────────────┐
│  surveillance_storage/   │  ← Storage directory
│  • Reads metadata JSON   │
│  • Serves frame images   │
│  • Searches reports      │
└──────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          API ENDPOINTS                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  GET  /sessions                    → List all sessions                  │
│  GET  /sessions/{id}/frames        → Get frames (paginated)             │
│  GET  /sessions/{id}/frames/{fid}  → Get frame image                    │
│  GET  /sessions/{id}/detections    → Get detection data                 │
│  GET  /sessions/{id}/tracks        → Get tracking data                  │
│  GET  /sessions/{id}/alerts        → Get alerts                         │
│  GET  /sessions/{id}/report        → Get VLM report                     │
│  POST /search                      → Semantic search                    │
│  GET  /health                      → Health check                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW DIAGRAM                                │
└─────────────────────────────────────────────────────────────────────────┘

Video → [FrameExtractor] → Frames → [Detector] → Detections
                              ↓
                        Saves to:                    ↓
                      raw_frame/              [Tracker] → Tracks
                   frames_metadata.json                  ↓
                   session_metadata.json           [AlertGen] → Alerts
                                                         ↓
                                                   Saves all to:
                                                   metadata/*.json
                                                         ↓
                                                   [Dashboard] reads
                                                         ↓
                                                   Frontend displays

Parallel VLM:
Video → [VLMNode] → Report → report.json + vectors/*.jsonl

┌─────────────────────────────────────────────────────────────────────────┐
│                      NODE COMMUNICATION TYPES                           │
└─────────────────────────────────────────────────────────────────────────┘

VIDEO             str or cv2.VideoCapture
                  ↓
FRAMES_BATCH      List[np.ndarray] or np.ndarray(N,H,W,C)
                  ↓
FRAME_META_BATCH  List[dict] with session_id, frame_id, timestamp, path
                  ↓
CLASS_BATCH       List[List[str]] - ["person", "car"] per frame
                  ↓
BBOX_BATCH        List[List[List[int]]] - [[x1,y1,x2,y2], ...] per frame
                  ↓
CONF_BATCH        List[List[float]] - [0.92, 0.87, ...] per frame
                  ↓
TRACKS_BATCH      List[List[dict]] - track_id, cls, bbox, conf per frame
                  ↓
ALERTS            List[dict] - alert_id, timestamp, reason, severity
                  ↓
REPORT            dict - comprehensive_report, summary, risk_level

┌─────────────────────────────────────────────────────────────────────────┐
│                        ALERT RULES ENGINE                               │
└─────────────────────────────────────────────────────────────────────────┘

Input: CLASS_BATCH + BBOX_BATCH + CONF_BATCH + TRACKS_BATCH
       │
       ├─→ Rule 1: Loitering Detection
       │   ├─ Track person position over time
       │   ├─ If stays in small area > threshold seconds
       │   └─ Generate ALERT (severity: Medium)
       │
       ├─→ Rule 2: Suspicious Object Detection
       │   ├─ Check for weapons in watchlist
       │   ├─ If detected with conf > threshold
       │   └─ Generate ALERT (severity: Critical)
       │
       └─→ Rule 3: Crowding Detection
           ├─ Count persons in frame
           ├─ If count > threshold
           └─ Generate ALERT (severity: Medium)

Output: ALERTS → alert.json

┌─────────────────────────────────────────────────────────────────────────┐
│                     EXAMPLE USAGE FLOW                                  │
└─────────────────────────────────────────────────────────────────────────┘

1. User runs: python example_pipeline.py video.mp4
   │
   ├─ FrameExtractor: Extracts 600 frames (stride=2)
   ├─ Detector: Finds 1,234 objects
   ├─ Tracker: Creates 45 unique tracks
   ├─ AlertGen: Generates 3 alerts (1 loitering, 2 crowding)
   └─ VLM: Analyzes video, risk level: Medium
   
2. System saves to: surveillance_storage/session_abc123/

3. User starts: python dashboard/backend.py
   │
   └─ Server runs on http://localhost:8000

4. User opens: dashboard/frontend.html
   │
   ├─ Fetches /sessions → Shows session_abc123
   ├─ Clicks session → Loads frames, alerts, report
   ├─ Navigates frames → GET /sessions/abc123/frames/f1234
   └─ Searches "loitering" → POST /search

5. Real-time updates: Alerts panel polls every 5 seconds

┌─────────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT OPTIONS                               │
└─────────────────────────────────────────────────────────────────────────┘

Development:
  python example_pipeline.py video.mp4
  python dashboard/backend.py
  Open dashboard/frontend.html

Production:
  uvicorn dashboard.backend:app --host 0.0.0.0 --port 8000
  Serve frontend.html via nginx/apache
  Set SURVEILLANCE_ROOT=/production/storage

Docker:
  docker build -t surveillance-system .
  docker run -p 8000:8000 \
    -v ./input:/app/input \
    -v ./surveillance_storage:/app/surveillance_storage \
    -e GEMINI_API_KEY=xxx \
    surveillance-system

┌─────────────────────────────────────────────────────────────────────────┐
│                     PERFORMANCE CHARACTERISTICS                         │
└─────────────────────────────────────────────────────────────────────────┘

Pipeline Mode     Stride  Speed    Accuracy   Use Case
──────────────────────────────────────────────────────────────────────────
Basic             1       Slow     Maximum    Post-analysis, Evidence
Fast              5       5x       Good       Real-time, Screening
Custom            N       N/x      Variable   Configurable

Example:
  30 FPS video, 10 minutes = 18,000 frames
  
  Stride=1:  18,000 frames processed (100% coverage)
  Stride=2:   9,000 frames processed (50% coverage, 2x faster)
  Stride=5:   3,600 frames processed (20% coverage, 5x faster)

Note: VLM always analyzes full video regardless of stride
```
