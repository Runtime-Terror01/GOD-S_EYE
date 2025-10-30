# NODE-BASED AI-POWERED VIDEO ANALYSIS SYSTEM FOR NSG

**Runtime Terror01** | **Smart India Hackathon 2025**

A modular, node-based platform for designing, deploying, and running custom video analysis pipelines for surveillance and security operations.

---

## 🎯 Overview

This system provides a **drag-and-drop node interface** to build custom video analysis pipelines tailored for National Security Guard (NSG) missions. It combines Computer Vision, Natural Language Processing, and Vector Database technologies to enable real-time analysis and semantic search of surveillance footage.

### Key Features

- **🎨 Modular Node-Based Platform**: Drag and drop nodes in an interactive interface to design custom video analysis pipelines
- **🖥️ Dual Frontend System**:
  - **Pipeline Builder**: ComfyUI-based visual workflow editor for mission-specific pipeline design
  - **Dashboard**: Real-time alerts, search, trends, and semantic chatbot interface
- **🤖 AI-Powered Analysis**: 
  - VLM (Vision-Language Models) for scene understanding
  - Vector Database for semantic search capabilities
  - Automated report generation
- **🔌 Pluggable Node Architecture**: Extensible nodes for Detection, Tracking, Face Recognition, Alert Generation, and VLM Analysis
- **📦 Central Repository**: Standardized storage format for processed data, enabling advanced analysis, model training, and research

---

## 📸 Screenshots

### Pipeline Builder (Node-Based Interface)
![Pipeline](screenshots\Pipeline.jpg)

### Surveillance Dashboard
![Dashboard](screenshots\Dashboard.jpg)

---

## 🏗️ Technical Architecture

### Frontend
- **ComfyUI** - Node-based pipeline designer with Litegraph
- **React.js, CSS, Tailwind** - Dashboard interface
- **HTML5** - Web components

### Backend
- **Python** - Core processing logic
- **FastAPI** - REST API server
- **LangChain** - LLM orchestration
- **ComfyUI Backend Server** - Node execution engine

### AI Models
- **YOLO** (YOLOv8) - Object Detection
- **ByteTrack** - Multi-object Tracking
- **ArcFace** - Face Recognition
- **Gemini** (Vision-Language Model) - Scene Understanding & Report Generation
- **Ollama** - Local LLM support for chatbot

### Central Repository
- **FastAPI** - Data access API
- **MongoDB** - Metadata storage (optional)
- **ChromaDB** - Vector database for semantic search
- **File System** - Structured storage for frames and metadata

### Deployment
- **Docker** - Containerization
- **On-Premise/Edge** - NSG system compatibility

---

## 🧩 Available Nodes

### CV Nodes (Computer Vision)
1. **Frame Extractor Node** - Extract frames from video sources (MP4/RTSP/drone feeds)
2. **Object Detection Node** - Detect objects using YOLOv8
3. **Tracking Node** - Track objects across frames using ByteTrack
4. **Face Recognition Node** - Identify faces using ArcFace
5. **Segmentation Node** - Semantic/instance segmentation

### NLP Nodes (Natural Language Processing)
1. **VLM Analysis Node** - Analyze video content using Vision-Language Models (Gemini)
2. **Semantic Search Node** - Search footage using natural language queries
3. **Alert Generation Node** - Generate alerts based on suspicious activity
4. **Report Generation Node** - Create comprehensive security reports

### Database Nodes
1. **Vector DB Node** - Store and query vector embeddings (ChromaDB)
2. **NoSQL DB Node** - Store metadata in MongoDB/SQLite

---

## 📁 Central Repository Structure

```
surveillance_storage/
├── session_<session_id>/
│   ├── raw_frame/
│   │   ├── frame_000000.jpg
│   │   ├── frame_000001.jpg
│   │   ├── frame_000002.jpg
│   │   └── ...
│   ├── annotated_frame/
│   │   ├── frame_000000.jpg
│   │   ├── frame_000001.jpg
│   │   └── ...
│   └── metadata/
│       ├── session_metadata.json
│       ├── frames_metadata.json
│       ├── detection_metadata.json
│       ├── tracking_metadata.json
│       ├── alert.json
│       ├── report.json
│       └── vlm_alert.json
│
├── session_<another_session_id>/   
├── ...
└── vectors/
        ├── embeddings.jsonl
        └── chroma.db
```

---

## 📋 Standardized Metadata Formats

### 1. session_metadata.json
```json
{
  "session_id": "d71c1d59",
  "video_path": "input/my_video.mp4",
  "started_at": "2025-10-14T09:00:00+05:30",
  "frame_count": 1200,
  "fps": 30,
  "resolution": "1920x1080"
}
```

### 2. frames_metadata.json
```json
[
  {
    "session_id": "d71c1d59",
    "frame_id": "f01fb80b-0e78-44ab-bfb4-69a26da19ecd",
    "frame_index": 0,
    "timestamp_sec": 0.0,
    "path": "session_d71c1d59/raw_frame/frame_000000.jpg"
  },
  {
    "session_id": "d71c1d59",
    "frame_id": "67774649-89cd-41f8-b0c8-560de2d71acc",
    "frame_index": 1,
    "timestamp_sec": 0.04,
    "path": "session_d71c1d59/raw_frame/frame_000001.jpg"
  }
]
```

### 3. detection_metadata.json
```json
{
  "session_id": "d71c1d59",
  "detections": {
    "frame_000000": [
      {"cls": "person", "bbox": [100, 150, 230, 400], "conf": 0.92},
      {"cls": "car", "bbox": [300, 200, 500, 450], "conf": 0.87}
    ],
    "frame_000001": [
      {"cls": "person", "bbox": [105, 152, 235, 402], "conf": 0.93}
    ]
  }
}
```

### 4. tracking_metadata.json
```json
{
  "session_id": "d71c1d59",
  "tracks": {
    "frame_000000": [
      {"track_id": 1, "cls": "person", "bbox": [100, 150, 230, 400], "conf": 0.92},
      {"track_id": 2, "cls": "car", "bbox": [300, 200, 500, 450], "conf": 0.87}
    ],
    "frame_000001": [
      {"track_id": 1, "cls": "person", "bbox": [105, 152, 235, 402], "conf": 0.93},
      {"track_id": 2, "cls": "car", "bbox": [305, 202, 505, 452], "conf": 0.88}
    ]
  }
}
```

### 5. alert.json
```json
{
  "session_id": "d71c1d59",
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

### 6. report.json
```json
{
  "session_id": "d71c1d59",
  "comprehensive_report": "### COMPREHENSIVE REPORT ###\nTIMESTAMP: 00:00:05 - 00:00:08\nOBSERVATIONS: ...\nFLAGGED_ITEMS: ...\n---\n",
  "summary_for_user": {
    "overall_risk_level": "Medium",
    "justification": "Short paragraph explaining the risk assessment...",
    "recommended_actions": ["Action 1", "Action 2"]
  },
  "generated_at": "2025-10-14T09:30:00+05:30"
}
```

### 7. vlm_alert.json
```json
{
  "session_id": "b628d100",
  "alerts": [
    {
      "alert_id": "alert_0000",
      "timestamp_sec": 0.0,
      "frame_ref": "time_00:00:00 - 00:00:08",
      "reason": "suspicious_activity",
      "evidence": {
        "flagged_items": "Military vehicles, Improvised Explosive Device (IED), small arms fire, Rocket Propelled Grenade (RPG), armed combatants.",
        "event_description": "TIMESTAMP: 00:00:00 - 00:00:08\nOBSERVATIONS: An aerial view shows a convoy of at least three military-style trucks driving on a winding dirt road through a dense forest."
      },
      "severity": "Critical"
    },
    {
      "alert_id": "alert_0001",
      "timestamp_sec": 9.0,
      "frame_ref": "time_00:00:09 - 00:00:12",
      "reason": "suspicious_activity",
      "evidence": {
        "flagged_items": "None.",
        "event_description": "TIMESTAMP: 00:00:09 - 00:00:12\nOBSERVATIONS: The scene shifts to a government building with an Indian flag."
      },
      "severity": "Critical"
    }
  ]
}
```

---

## 🔌 Dashboard API Specification

### Backend (FastAPI)

**Configuration**: Set `SURVEILLANCE_ROOT` environment variable to point to `surveillance_storage/` directory.

**Server**: FastAPI with Uvicorn

**CORS**: Enabled for frontend development

### API Endpoints

#### Session Management
```
GET /sessions
```
Returns list of all session IDs with their metadata.

**Response**:
```json
[
  {
    "session_id": "d71c1d59",
    "video_path": "input/my_video.mp4",
    "started_at": "2025-10-14T09:00:00+05:30",
    "frame_count": 1200,
    "fps": 30
  }
]
```

#### Frame Metadata
```
GET /sessions/{session_id}/frames?start=0&end=100&limit=50
```
Returns paginated frame metadata.

**Response**: Array from `frames_metadata.json`

#### Frame Image
```
GET /sessions/{session_id}/frames/{frame_id}
```
Returns the actual frame image file.

**Response**: JPEG image file

#### Detections
```
GET /sessions/{session_id}/detections
```
Returns detection metadata for all frames.

**Response**: Content of `detection_metadata.json`

#### Tracking
```
GET /sessions/{session_id}/tracks
```
Returns tracking metadata for all frames.

**Response**: Content of `tracking_metadata.json`

#### Alerts
```
GET /sessions/{session_id}/alerts
```
Returns all alerts generated for the session.

**Response**: Content of `alert.json`

#### Report
```
GET /sessions/{session_id}/report
```
Returns comprehensive security report.

**Response**: Content of `report.json`

#### Semantic Search
```
POST /search
```
Performs semantic search across all sessions using vector database.

**Request Body**:
```json
{
  "query": "Show me footage with suspicious vehicles",
  "limit": 10
}
```

**Response**:
```json
{
  "results": [
    {
      "session_id": "d71c1d59",
      "timestamp_sec": 45.2,
      "relevance_score": 0.89,
      "snippet": "Military-style vehicle observed at perimeter..."
    }
  ]
}
```

---

## 🖥️ Dashboard Frontend Features

### Minimal Frontend Components

1. **Session List** - Displays all available surveillance sessions
2. **Frame Viewer** - Shows individual frames with overlays
3. **Alerts Panel** - Real-time alert notifications (polling)
4. **Semantic Chatbot** - Natural language query interface for searching footage
5. **Timeline View** - Visual timeline of events and detections
6. **Face Match List** - Track identified individuals
7. **Trend Analysis** - Statistical overview of detected objects/events

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.10+
python --version

# CUDA (for GPU acceleration)
nvidia-smi
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd GOD-S_EYE

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_surveillance.txt

# Set up API keys
# Create .env file with:
# GOOGLE_API_KEY=your_gemini_api_key
```

### Running the System

#### 1. Start Pipeline Builder (ComfyUI)
```bash
python main.py
```
Access at: `http://localhost:8188`

#### 2. Start Dashboard Backend
```bash
cd dashboard
uvicorn backend:app --reload --port 8000
```
Access API at: `http://localhost:8000`

#### 3. Start Dashboard Frontend
Open `dashboard/frontend.html` in browser or serve with:
```bash
# If using React dashboard
cd surveillance-dashboard
npm install
npm start
```
Access at: `http://localhost:3000`

---

## 📦 Project Structure

```
GOD-S_EYE/
├── custom_nodes/              # Custom surveillance nodes
│   └── nodes/
│       ├── frame_extractor_node.py
│       ├── object_detection_node.py
│       ├── tracking_node.py
│       ├── vlm_analysis_node.py
│       ├── alert_node.py
│       ├── data_types.py      # Standardized data type definitions
│       └── base_node.py
├── dashboard/                 # Dashboard backend & frontend
│   ├── backend.py            # FastAPI server
│   └── frontend.html         # Minimal HTML dashboard
├── surveillance-dashboard/    # React dashboard (optional)
├── surveillance_storage/      # Central data repository
├── input/                    # Input videos
├── models/                   # AI model weights
├── main.py                   # ComfyUI server entry point
└── requirements.txt
```

---

## 🔧 System Capabilities

### Input Flexibility
- ✅ MP4/AVI/MKV video files
- ✅ RTSP live streams
- ✅ Drone camera feeds
- ✅ Bodycam footage
- ✅ Legacy surveillance devices (no hardware upgrade required)

### Real-Time Processing
- ✅ Process massive video data in real-time
- ✅ Analyze content impossible for humans to review manually
- ✅ Flexible pipeline modification for mission-specific needs

### Future-Ready Architecture
- ✅ Modular node design for easy feature addition
- ✅ Vision + Language model integration
- ✅ Extendable with custom nodes
- ✅ Data ready for model training and research

---

## 🛠️ Development

### Creating Custom Nodes

1. Inherit from `BaseNode` class
2. Define `INPUT_TYPES` and `RETURN_TYPES`
3. Implement `process()` method
4. Use standardized data types from `data_types.py`

**Example**:
```python
from custom_nodes.nodes.base_node import BaseNode
from custom_nodes.nodes.data_types import FRAMES_BATCH, FRAME_META_BATCH

class MyCustomNode(BaseNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("FRAMES_BATCH",),
                "frame_metadata": ("FRAME_META_BATCH",)
            }
        }
    
    RETURN_TYPES = ("FRAMES_BATCH",)
    FUNCTION = "process"
    CATEGORY = "surveillance"
    
    def process(self, frames, frame_metadata):
        # Your processing logic
        return (processed_frames,)
```

---

## 📄 License

[Add license information]

---

## 👥 Team

**Runtime Terror01** - Smart India Hackathon 2025

---

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines first.

---

## 📞 Support

For issues and questions, please open an issue on the repository.

---

**Built for NSG | Smart India Hackathon 2025**