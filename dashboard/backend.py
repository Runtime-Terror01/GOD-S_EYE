"""
Dashboard Backend - FastAPI Server
Serves surveillance data through RESTful API
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
SURVEILLANCE_ROOT = os.getenv('SURVEILLANCE_ROOT', './surveillance_storage')

app = FastAPI(
    title="Surveillance Dashboard API",
    description="API for accessing surveillance system data",
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Models ====================

class SessionInfo(BaseModel):
    session_id: str
    video_path: str
    started_at: str
    frame_count: int
    fps: float


class SearchQuery(BaseModel):
    query: str
    limit: int = 10


# ==================== Helper Functions ====================

def get_session_dir(session_id: str) -> Path:
    """Get path to session directory"""
    return Path(SURVEILLANCE_ROOT) / f"session_{session_id}"


def get_metadata_dir(session_id: str) -> Path:
    """Get path to metadata directory"""
    return get_session_dir(session_id) / "metadata"


def load_json_file(filepath: Path) -> Dict:
    """Load JSON file, return empty dict if not found or invalid"""
    if not filepath.exists():
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.exception("Failed to decode JSON file: %s", filepath)
        return {}
    except Exception:
        logger.exception("Failed to read file: %s", filepath)
        return {}


def list_sessions() -> List[str]:
    """List all available session IDs"""
    root = Path(SURVEILLANCE_ROOT)
    if not root.exists():
        return []
    sessions = []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith('session_'):
            session_id = item.name.replace('session_', '')
            sessions.append(session_id)
    return sorted(sessions)


# ==================== Endpoints ====================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Surveillance Dashboard API",
        "version": "1.0.0",
        "endpoints": [
            "/sessions",
            "/sessions/{session_id}/frames",
            "/sessions/{session_id}/frames/{frame_id}",
            "/sessions/{session_id}/detections",
            "/sessions/{session_id}/tracks",
            "/sessions/{session_id}/alerts",
            "/sessions/{session_id}/report",
            "/search"
        ]
    }


@app.get("/sessions", response_model=List[SessionInfo])
async def get_sessions():
    """
    List all surveillance sessions with metadata
    """
    sessions = []
    for session_id in list_sessions():
        metadata_dir = get_metadata_dir(session_id)
        session_meta_file = metadata_dir / "session_metadata.json"

        if session_meta_file.exists():
            session_meta = load_json_file(session_meta_file)
            # Ensure all keys exist; provide defaults if missing
            session_meta_defaults = {
                "session_id": session_id,
                "video_path": session_meta.get("video_path", "unknown"),
                "started_at": session_meta.get("started_at", datetime.now().isoformat()),
                "frame_count": int(session_meta.get("frame_count", 0)),
                "fps": float(session_meta.get("fps", 0.0))
            }
            sessions.append(SessionInfo(**session_meta_defaults))
        else:
            sessions.append(SessionInfo(
                session_id=session_id,
                video_path="unknown",
                started_at=datetime.now().isoformat(),
                frame_count=0,
                fps=0.0
            ))
    return sessions


@app.get("/sessions/{session_id}/frames")
async def get_frames(
    session_id: str,
    start: int = Query(0, ge=0, description="Start index"),
    end: Optional[int] = Query(None, description="End index"),
    limit: int = Query(100, ge=1, description="Maximum number of frames")
):
    """
    Get paginated frames metadata for a session
    """
    metadata_dir = get_metadata_dir(session_id)
    frames_meta_file = metadata_dir / "frames_metadata.json"

    if not frames_meta_file.exists():
        raise HTTPException(status_code=404, detail=f"Frames metadata not found for session {session_id}")

    frames_meta = load_json_file(frames_meta_file)
    if not isinstance(frames_meta, list):
        raise HTTPException(status_code=500, detail="Invalid frames metadata format")

    total = len(frames_meta)
    if end is None:
        end = start + limit
    # Clamp indices
    start = max(0, min(start, total))
    end = max(0, min(end, total))
    if end < start:
        end = start

    paginated = frames_meta[start:end]

    return {
        "session_id": session_id,
        "total_frames": total,
        "start": start,
        "end": end,
        "frames": paginated
    }


@app.get("/sessions/{session_id}/frames/{frame_id}")
async def get_frame_image(session_id: str, frame_id: str):
    """
    Get frame image file by frame_id (uuid) or by filename (frame_000000.jpg)
    """
    metadata_dir = get_metadata_dir(session_id)
    frames_meta_file = metadata_dir / "frames_metadata.json"

    if not frames_meta_file.exists():
        raise HTTPException(status_code=404, detail=f"Frames metadata not found for session {session_id}")

    frames_meta = load_json_file(frames_meta_file)
    if not isinstance(frames_meta, list):
        raise HTTPException(status_code=500, detail="Invalid frames metadata format")

    # Find frame by frame_id OR by filename if user passed filename
    frame_info = None
    for frame in frames_meta:
        if frame.get('frame_id') == frame_id or Path(frame.get('path', '')).name == frame_id:
            frame_info = frame
            break

    if not frame_info:
        raise HTTPException(status_code=404, detail=f"Frame {frame_id} not found")

    frame_path = Path(SURVEILLANCE_ROOT) / frame_info['path']
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame image file not found: {frame_path}")

    return FileResponse(frame_path)


@app.get("/sessions/{session_id}/detections")
async def get_detections(session_id: str):
    """
    Get detection metadata for a session
    """
    metadata_dir = get_metadata_dir(session_id)
    detection_file = metadata_dir / "detection_metadata.json"

    if not detection_file.exists():
        raise HTTPException(status_code=404, detail=f"Detection metadata not found for session {session_id}")

    return load_json_file(detection_file)


@app.get("/sessions/{session_id}/tracks")
async def get_tracks(session_id: str):
    """
    Get tracking metadata for a session
    """
    metadata_dir = get_metadata_dir(session_id)
    tracking_file = metadata_dir / "tracking_metadata.json"

    if not tracking_file.exists():
        raise HTTPException(status_code=404, detail=f"Tracking metadata not found for session {session_id}")

    return load_json_file(tracking_file)


@app.get("/sessions/{session_id}/alerts")
async def get_alerts(session_id: str):
    """
    Get alerts for a session
    """
    metadata_dir = get_metadata_dir(session_id)
    alert_file = metadata_dir / "alert.json"

    if not alert_file.exists():
        return {"session_id": session_id, "alerts": []}

    data = load_json_file(alert_file)
    # if malformed, return empty
    if not isinstance(data, dict):
        return {"session_id": session_id, "alerts": []}
    return data


@app.get("/sessions/{session_id}/report")
async def get_report(session_id: str):
    """
    Get VLM report for a session
    """
    metadata_dir = get_metadata_dir(session_id)
    report_file = metadata_dir / "report.json"

    if not report_file.exists():
        raise HTTPException(status_code=404, detail=f"Report not found for session {session_id}")

    return load_json_file(report_file)


@app.post("/search")
async def search(query: SearchQuery):
    """
    Semantic search across reports and session data (simple keyword match fallback)
    """
    results = []
    q_lower = query.query.lower().strip()

    for session_id in list_sessions():
        metadata_dir = get_metadata_dir(session_id)
        report_file = metadata_dir / "report.json"
        if not report_file.exists():
            continue
        report = load_json_file(report_file)
        if not isinstance(report, dict):
            continue

        comprehensive = report.get('comprehensive_report', '').lower()
        summary = json.dumps(report.get('summary_for_user', {})).lower()

        if q_lower in comprehensive or q_lower in summary:
            results.append({
                "session_id": session_id,
                "relevance": "high" if q_lower in summary else "medium",
                "snippet": report.get('summary_for_user', {}).get('justification', '')[:200],
                "risk_level": report.get('summary_for_user', {}).get('overall_risk_level', 'Unknown'),
                "generated_at": report.get('generated_at', report.get('generated_at', ''))
            })
        if len(results) >= query.limit:
            break

    return {
        "query": query.query,
        "total_results": len(results),
        "results": results
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "surveillance_root": SURVEILLANCE_ROOT,
        "sessions_count": len(list_sessions())
    }


# ==================== Main ====================

if __name__ == "__main__":
    Path(SURVEILLANCE_ROOT).mkdir(parents=True, exist_ok=True)
    logger.info("Starting Dashboard Backend...")
    logger.info("Surveillance Root: %s", SURVEILLANCE_ROOT)
    logger.info("Server: http://localhost:8000")
    logger.info("API Docs: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
