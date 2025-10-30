# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from pathlib import Path
# import json
# from typing import Dict, List, Optional
# import os
# from collections import Counter
# import numpy as np

# app = FastAPI()

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# STORAGE_PATH = Path("surveillance_storage")

# def load_json_file(filepath: Path) -> Optional[Dict]:
#     """Load and parse JSON file safely"""
#     try:
#         if filepath.exists():
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 return json.load(f)
#     except Exception as e:
#         print(f"Error loading {filepath}: {e}")
#     return None

# @app.get("/api/sessions")
# async def get_sessions():
#     """Get all available session folders"""
#     try:
#         if not STORAGE_PATH.exists():
#             return {"sessions": []}
        
#         sessions = []
#         for session_dir in STORAGE_PATH.iterdir():
#             if session_dir.is_dir() and session_dir.name.startswith("session_"):
#                 session_id = session_dir.name.replace("session_", "")
#                 metadata_path = session_dir / "metadata" / "session_metadata.json"
#                 metadata = load_json_file(metadata_path)
                
#                 sessions.append({
#                     "session_id": session_id,
#                     "metadata": metadata or {},
#                     "path": str(session_dir)
#                 })
        
#         return {"sessions": sorted(sessions, key=lambda x: x["session_id"], reverse=True)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/session/{session_id}")
# async def get_session_data(session_id: str):
#     """Get complete session data including all metadata"""
#     try:
#         session_path = STORAGE_PATH / f"session_{session_id}"
#         if not session_path.exists():
#             raise HTTPException(status_code=404, detail="Session not found")
        
#         metadata_path = session_path / "metadata"
        
#         # Load all metadata files
#         session_metadata = load_json_file(metadata_path / "session_metadata.json") or {}
#         frames_metadata = load_json_file(metadata_path / "frames_metadata.json") or []
#         detection_metadata = load_json_file(metadata_path / "detection_metadata.json") or {}
#         tracking_metadata = load_json_file(metadata_path / "tracking_metadata.json") or {}
#         alerts = load_json_file(metadata_path / "alert.json") or {"alerts": []}
#         report = load_json_file(metadata_path / "report.json") or {}
#         vlm_report = load_json_file(metadata_path / "vlm_report.json") or {}
        
#         # Count frames
#         raw_frame_path = session_path / "raw_frame"
#         annotated_frame_path = session_path / "annotated_frame"
        
#         raw_frames = sorted(raw_frame_path.glob("*.jpg")) if raw_frame_path.exists() else []
#         annotated_frames = sorted(annotated_frame_path.glob("*.jpg")) if annotated_frame_path.exists() else []
        
#         # Add fake alerts if none exist
#         alert_list = alerts.get("alerts", [])
#         if not alert_list:
#             alert_list = [
#                 {
#                     "reason": "Suspicious Movement Pattern",
#                     "severity": "Critical",
#                     "timestamp_sec": 15.3,
#                     "frame_ref": "frame_000458",
#                     "confidence": 0.89
#                 },
#                 {
#                     "reason": "Unidentified Personnel",
#                     "severity": "High", 
#                     "timestamp_sec": 42.7,
#                     "frame_ref": "frame_001281",
#                     "confidence": 0.76
#                 },
#                 {
#                     "reason": "Weapon Detection",
#                     "severity": "Critical",
#                     "timestamp_sec": 78.2,
#                     "frame_ref": "frame_002346",
#                     "confidence": 0.94
#                 },
#                 {
#                     "reason": "Perimeter Breach",
#                     "severity": "Medium",
#                     "timestamp_sec": 105.8,
#                     "frame_ref": "frame_003174",
#                     "confidence": 0.82
#                 },
#                 {
#                     "reason": "Vehicle Anomaly",
#                     "severity": "High",
#                     "timestamp_sec": 134.5,
#                     "frame_ref": "frame_004035",
#                     "confidence": 0.71
#                 }
#             ]

#         return {
#             "session_id": session_id,
#             "session_metadata": session_metadata,
#             "frames_metadata": frames_metadata,
#             "detection_metadata": detection_metadata,
#             "tracking_metadata": tracking_metadata,
#             "alerts": alert_list,
#             "report": report,
#             "vlm_report": vlm_report,
#             "frame_count": len(raw_frames),
#             "annotated_frame_count": len(annotated_frames)
#         }
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/session/{session_id}/frame/{frame_index}")
# async def get_frame(session_id: str, frame_index: int, annotated: bool = True):
#     """Get specific frame image"""
#     try:
#         session_path = STORAGE_PATH / f"session_{session_id}"
#         frame_dir = session_path / ("annotated_frame" if annotated else "raw_frame")
        
#         frame_file = frame_dir / f"frame_{frame_index:06d}.jpg"
        
#         if not frame_file.exists():
#             raise HTTPException(status_code=404, detail="Frame not found")
        
#         return FileResponse(frame_file, media_type="image/jpeg")
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/session/{session_id}/visualizations")
# async def generate_visualizations(session_id: str):
#     """Generate matplotlib-based visualization chart"""
#     try:
#         from chart_generator import visualize_session_three_plots
        
#         session_path = STORAGE_PATH / f"session_{session_id}"
        
#         # Generate the chart and return as base64
#         chart_base64 = visualize_session_three_plots(
#             str(session_path), 
#             layout='vertical', 
#             return_base64=True
#         )
        
#         return {"chart_image": chart_base64}
#     except Exception as e:
#         # Return dummy chart if generation fails
#         import base64
#         from io import BytesIO
#         import matplotlib.pyplot as plt
#         import numpy as np
        
#         plt.style.use('dark_background')
#         fig, axes = plt.subplots(3, 1, figsize=(12, 14))
#         fig.patch.set_facecolor('#0f0f0f')
#         fig.suptitle(f"Security Analysis — Session {session_id}", fontsize=16, weight='bold', color='white')
        
#         # Dummy detection chart
#         classes = ['person', 'vehicle', 'backpack', 'weapon', 'phone']
#         counts = [45, 23, 12, 8, 15]
#         y_pos = np.arange(len(classes))[::-1]
#         axes[0].barh(y_pos, counts, color='steelblue', edgecolor='black')
#         axes[0].set_yticks(y_pos)
#         axes[0].set_yticklabels(classes, color='white')
#         axes[0].set_xlabel("Count", color='white')
#         axes[0].set_title("Detected Classes", color='white', fontsize=12)
#         axes[0].set_facecolor('#1a1a1a')
#         axes[0].tick_params(colors='white')
        
#         # Dummy activity chart
#         time_points = np.linspace(0, 120, 50)
#         activity = np.random.poisson(8, 50) + np.sin(time_points/10) * 3 + 5
#         axes[1].plot(time_points, activity, color='orangered', linewidth=2)
#         axes[1].set_xlabel("Time (seconds)", color='white')
#         axes[1].set_ylabel("Object Count", color='white')
#         axes[1].set_title("Activity Level Over Time", color='white', fontsize=12)
#         axes[1].set_facecolor('#1a1a1a')
#         axes[1].tick_params(colors='white')
#         axes[1].grid(True, alpha=0.3)
        
#         # Dummy track chart
#         durations = np.random.exponential(15, 25)
#         axes[2].hist(durations, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
#         axes[2].set_xlabel("Track Duration (seconds)", color='white')
#         axes[2].set_ylabel("Number of Tracks", color='white')
#         axes[2].set_title("Track Duration Summary", color='white', fontsize=12)
#         axes[2].set_facecolor('#1a1a1a')
#         axes[2].tick_params(colors='white')
        
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
#         buffer = BytesIO()
#         plt.savefig(buffer, format='png', facecolor='#0f0f0f', dpi=150, bbox_inches='tight')
#         buffer.seek(0)
#         chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
#         plt.close()
        
#         return {"chart_image": f"data:image/png;base64,{chart_base64}"}

# @app.get("/api/session/{session_id}/detected_faces")
# async def get_detected_faces(session_id: str):
#     """Get detected faces data with real dummy images"""
#     try:
#         # Return dummy faces with real image URLs
#         dummy_faces = [
#             {
#                 "id": "target_a", 
#                 "name": "Target Alpha", 
#                 "confidence": 87, 
#                 "threat_level": "high",
#                 "image_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face"
#             },
#             {
#                 "id": "target_b", 
#                 "name": "Target Bravo", 
#                 "confidence": 92, 
#                 "threat_level": "critical",
#                 "image_url": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face"
#             },
#             {
#                 "id": "target_c", 
#                 "name": "Target Charlie", 
#                 "confidence": 78, 
#                 "threat_level": "medium",
#                 "image_url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150&h=150&fit=crop&crop=face"
#             },
#             {
#                 "id": "target_d", 
#                 "name": "Target Delta", 
#                 "confidence": 95, 
#                 "threat_level": "critical",
#                 "image_url": "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=150&h=150&fit=crop&crop=face"
#             },
#             {
#                 "id": "target_e", 
#                 "name": "Target Echo", 
#                 "confidence": 83, 
#                 "threat_level": "high",
#                 "image_url": "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=150&h=150&fit=crop&crop=face"
#             }
#         ]
        
#         return {"faces": dummy_faces}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/search")
# async def semantic_search(query: dict):
#     """Handle semantic search queries with intelligent responses"""
#     try:
#         search_text = query.get("query", "").lower()
#         session_id = query.get("session_id")
        
#         if not session_id:
#             return {"response": "Please select a session first."}
        
#         session_path = STORAGE_PATH / f"session_{session_id}"
#         metadata_path = session_path / "metadata"
        
#         # Load VLM report data
#         vlm_report = load_json_file(metadata_path / "vlm_report.json") or {}
#         detection_metadata = load_json_file(metadata_path / "detection_metadata.json") or {}
#         tracking_metadata = load_json_file(metadata_path / "tracking_metadata.json") or {}
#         session_metadata = load_json_file(metadata_path / "session_metadata.json") or {}
        
#         # Get recommended actions from VLM report
#         summary = vlm_report.get("summary_for_user", {})
#         recommended_actions = summary.get("recommended_actions", [])
#         risk_level = summary.get("overall_risk_level", "Medium")
        
#         # Intelligent query processing
#         response = ""
        
#         if any(word in search_text for word in ["report", "summary", "overview", "analysis"]):
#             if recommended_actions:
#                 response = f"INTELLIGENCE ASSESSMENT - Session {session_id}\n\nRisk Level: {risk_level}\n\nRecommended Actions:\n"
#                 for i, action in enumerate(recommended_actions, 1):
#                     response += f"{i}. {action}\n"
#             else:
#                 response = f"Session {session_id} - Risk Level: {risk_level}. Detailed analysis in progress."
                
#         elif any(word in search_text for word in ["action", "recommend", "next", "steps"]):
#             if recommended_actions:
#                 response = "RECOMMENDED ACTIONS:\n"
#                 for i, action in enumerate(recommended_actions, 1):
#                     response += f"{i}. {action}\n"
#             else:
#                 response = "Standard surveillance protocols apply. Monitor for suspicious activity and maintain situational awareness."
                
#         elif any(word in search_text for word in ["threat", "risk", "danger", "security"]):
#             response = f"THREAT ASSESSMENT: {risk_level} risk level detected. "
#             if "critical" in risk_level.lower():
#                 response += "Immediate action required. Escalate to command authority."
#             elif "high" in risk_level.lower():
#                 response += "Enhanced monitoring protocols activated."
#             else:
#                 response += "Standard security measures in effect."
                
#         elif any(word in search_text for word in ["military", "tactical", "operation", "engagement"]):
#             if vlm_report.get("comprehensive_report"):
#                 response = "TACTICAL ANALYSIS: Military engagement detected in footage. Key observations include tactical movement patterns, equipment identification, and operational procedures. Recommend intelligence analysis for strategic assessment."
#             else:
#                 response = "No military activity detected in current session."
                
#         elif any(word in search_text for word in ["person", "people", "individual", "suspect"]):
#             # Dummy detection count
#             response = "PERSONNEL DETECTION: Identified 12 individuals in surveillance footage. 3 flagged for enhanced monitoring based on behavioral analysis."
            
#         elif any(word in search_text for word in ["vehicle", "transport", "convoy"]):
#             response = "VEHICLE ANALYSIS: 5 vehicles detected including 2 armored personnel carriers and 1 main battle tank. Movement patterns suggest coordinated operation."
            
#         elif any(word in search_text for word in ["weapon", "arms", "equipment"]):
#             response = "EQUIPMENT ASSESSMENT: Multiple weapons systems identified. Standard military loadout observed. No prohibited items detected outside operational parameters."
            
#         elif any(word in search_text for word in ["time", "duration", "when", "timeline"]):
#             response = f"TEMPORAL ANALYSIS: Session duration approximately 2 minutes 55 seconds. Key events occurred at timestamps 00:14, 01:12, and 02:30. Peak activity between 01:25-02:05."
            
#         elif any(word in search_text for word in ["location", "where", "area", "zone"]):
#             response = "LOCATION INTELLIGENCE: Urban environment with mixed residential/commercial structures. Arid climate zone. Multiple entry/exit points identified."
            
#         else:
#             # Default intelligent response
#             responses = [
#                 f"Analysis complete for Session {session_id}. Risk level: {risk_level}. Query processed through surveillance intelligence system.",
#                 "Surveillance data analyzed. No immediate threats detected. Continuing monitoring protocols.",
#                 "Intelligence assessment ongoing. All parameters within normal operational ranges.",
#                 f"Session {session_id} reviewed. Standard security protocols maintained. Risk assessment: {risk_level}."
#             ]
#             import random
#             response = random.choice(responses)
        
#         return {"response": response}
#     except Exception as e:
#         return {"response": "Intelligence system operational. Query processed successfully."}

# # Serve static files
# @app.get("/")
# async def serve_frontend():
#     return FileResponse("index.html")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import json
from typing import Dict, Optional
import base64
from io import BytesIO
from collections import Counter
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_PATH = Path("surveillance_storage")

def load_json_file(filepath: Path) -> Optional[Dict]:
    """Load and parse JSON file safely"""
    try:
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return None

@app.get("/api/sessions")
async def get_sessions():
    """Get all available session folders"""
    try:
        if not STORAGE_PATH.exists():
            return {"sessions": []}
        
        sessions = []
        for session_dir in STORAGE_PATH.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith("session_"):
                session_id = session_dir.name.replace("session_", "")
                metadata_path = session_dir / "metadata" / "session_metadata.json"
                metadata = load_json_file(metadata_path)
                
                sessions.append({
                    "session_id": session_id,
                    "metadata": metadata or {},
                    "path": str(session_dir)
                })
        
        return {"sessions": sorted(sessions, key=lambda x: x["session_id"], reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
async def get_session_data(session_id: str):
    """Get complete session data including all metadata"""
    try:
        session_path = STORAGE_PATH / f"session_{session_id}"
        if not session_path.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        metadata_path = session_path / "metadata"
        
        session_metadata = load_json_file(metadata_path / "session_metadata.json") or {}
        frames_metadata = load_json_file(metadata_path / "frames_metadata.json") or []
        detection_metadata = load_json_file(metadata_path / "detection_metadata.json") or {}
        tracking_metadata = load_json_file(metadata_path / "tracking_metadata.json") or {}
        alerts = load_json_file(metadata_path / "alert.json") or {"alerts": []}
        report = load_json_file(metadata_path / "report.json") or {}
        vlm_report = load_json_file(metadata_path / "vlm_report.json") or {}
        
        raw_frame_path = session_path / "raw_frame"
        annotated_frame_path = session_path / "annotated_frame"
        
        raw_frames = sorted(raw_frame_path.glob("*.jpg")) if raw_frame_path.exists() else []
        annotated_frames = sorted(annotated_frame_path.glob("*.jpg")) if annotated_frame_path.exists() else []
        
        return {
            "session_id": session_id,
            "session_metadata": session_metadata,
            "frames_metadata": frames_metadata,
            "detection_metadata": detection_metadata,
            "tracking_metadata": tracking_metadata,
            "alerts": alerts.get("alerts", []),
            "report": report,
            "vlm_report": vlm_report,
            "frame_count": len(raw_frames),
            "annotated_frame_count": len(annotated_frames)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}/frame/{frame_index}")
async def get_frame(session_id: str, frame_index: int, annotated: bool = True):
    """Get specific frame image"""
    try:
        session_path = STORAGE_PATH / f"session_{session_id}"
        frame_dir = session_path / ("annotated_frame" if annotated else "raw_frame")
        
        frame_file = frame_dir / f"frame_{frame_index:06d}.jpg"
        
        if not frame_file.exists():
            raise HTTPException(status_code=404, detail="Frame not found")
        
        return FileResponse(frame_file, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}/visualizations")
async def generate_visualizations(session_id: str):
    """Generate visualization charts from session data"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        
        session_path = STORAGE_PATH / f"session_{session_id}"
        metadata_path = session_path / "metadata"
        
        def load_session_data(session_path: Path) -> dict:
            data = {}
            metadata_path = session_path / "metadata"
            if not metadata_path.is_dir():
                raise FileNotFoundError(f"Metadata directory not found")
            for file in metadata_path.glob("*.json"):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        content = json.load(f)
                        data[file.stem] = content if content else None
                except (json.JSONDecodeError, TypeError):
                    data[file.stem] = None
            return data
        
        session_data = load_session_data(session_path)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(3, 1, figsize=(14, 14))
        fig.patch.set_facecolor('#1a1a1a')
        
        session_id_display = session_data.get("session_metadata", {}).get("session_id", session_id)
        fig.suptitle(f"Security Analysis - Session {session_id_display}", fontsize=16, weight='bold', color='white')
        
        # Plot 1: Detection Classes
        ax = axes[0]
        ax.set_facecolor('#1a1a1a')
        ax.set_title("Detected Classes (Grouped)", fontsize=12, color='white')
        
        detections = session_data.get("detection_metadata", {}).get("detections", {})
        if detections:
            class_counts = Counter(d.get("cls", "unknown") for dets in detections.values() for d in dets)
            total = sum(class_counts.values())
            
            if total > 0:
                items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:12]
                labels, counts = zip(*items)
                percents = [c / total * 100 for c in counts]
                y_pos = np.arange(len(labels))[::-1]
                
                ax.barh(y_pos, percents, edgecolor='black', color='steelblue')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=9, color='white')
                ax.set_xlabel("Percent of Detections (%)", color='white')
                ax.tick_params(colors='white')
                
                for i, p in enumerate(percents):
                    ax.text(p + 0.3, y_pos[i], f"{p:.1f}%", va='center', fontsize=8, color='white')
                ax.grid(axis='x', linestyle='--', alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No Detection Data", ha="center", va="center", color='white', transform=ax.transAxes)
        
        # Plot 2: Activity Timeline
        ax = axes[1]
        ax.set_facecolor('#1a1a1a')
        ax.set_title("Activity Level (Object Count Over Time)", fontsize=12, color='white')
        
        frame_meta = session_data.get("frames_metadata")
        if frame_meta and detections:
            index_to_time = {int(item["frame_index"]): float(item["timestamp_sec"]) for item in frame_meta}
            time_counts = []
            
            for frame_key, dets_in_frame in detections.items():
                try:
                    timestamp = index_to_time.get(int(frame_key.split("_")[-1]))
                    if timestamp is not None:
                        time_counts.append({"time": timestamp, "count": len(dets_in_frame)})
                except:
                    continue
            
            if time_counts:
                df = pd.DataFrame(time_counts).sort_values("time").reset_index(drop=True)
                ax.step(df["time"], df["count"], where='post', linewidth=0.8, alpha=0.6, label="Raw Count", color='gray')
                
                fps = session_data.get("session_metadata", {}).get("fps", 25)
                window = int(max(1, round(fps * 3)))
                df["smoothed"] = df["count"].rolling(window=window, min_periods=1, center=True).mean()
                ax.plot(df["time"], df["smoothed"], linewidth=2, label=f"Smoothed Trend", color="orangered")
                
                ax.set_xlabel("Time (seconds)", color='white')
                ax.set_ylabel("Number of Objects Detected", color='white')
                ax.tick_params(colors='white')
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(loc="upper right", fontsize=9, facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
        else:
            ax.text(0.5, 0.5, "No Temporal Data", ha="center", va="center", color='white', transform=ax.transAxes)
        
        # Plot 3: Track Duration
        ax = axes[2]
        ax.set_facecolor('#1a1a1a')
        ax.set_title("Track Duration Summary", fontsize=12, color='white')
        
        tracks = session_data.get("tracking_metadata", {}).get("tracks", {})
        fps = session_data.get("session_metadata", {}).get("fps", 1.0) or 1.0
        
        if tracks:
            track_lengths = Counter(track["track_id"] for frame_tracks in tracks.values() for track in frame_tracks)
            
            if track_lengths:
                durations_sec = np.array([frames / fps for frames in track_lengths.values()])
                
                if durations_sec.max() / max(durations_sec.min(), 1e-6) > 50:
                    bins = np.logspace(np.log10(max(durations_sec.min(), 0.1)), np.log10(durations_sec.max()), 30)
                    ax.set_xscale("log")
                    ax.set_xlabel("Track Duration (seconds, log scale)", color='white')
                else:
                    bins = np.arange(0, durations_sec.max() + 5, 5)
                    ax.set_xlabel("Track Duration (seconds)", color='white')
                
                ax.hist(durations_sec, bins=bins, color="skyblue", edgecolor="black")
                ax.set_ylabel("Number of Unique Tracks", color='white')
                ax.tick_params(colors='white')
                
                median, mean = np.median(durations_sec), np.mean(durations_sec)
                ax.axvline(median, color='red', linestyle='--', linewidth=1.5, label=f"Median: {median:.1f}s")
                ax.axvline(mean, color='green', linestyle=':', linewidth=1.5, label=f"Mean: {mean:.1f}s")
                ax.legend(fontsize=8, facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
            else:
                ax.text(0.5, 0.5, "No Tracks Found", ha="center", va="center", color='white', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No Tracking Data", ha="center", va="center", color='white', transform=ax.transAxes)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#1a1a1a', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {"chart_image": f"data:image/png;base64,{chart_base64}"}
        
    except Exception as e:
        print(f"Visualization error: {e}")
        return {"chart_image": None, "error": str(e)}

@app.post("/api/search")
async def semantic_search(query: dict):
    """Handle semantic search queries"""
    try:
        search_text = query.get("query", "").lower()
        session_id = query.get("session_id")
        
        if not session_id:
            return {"response": "Please select a session first."}
        
        session_path = STORAGE_PATH / f"session_{session_id}"
        metadata_path = session_path / "metadata"
        
        detection_metadata = load_json_file(metadata_path / "detection_metadata.json") or {}
        tracking_metadata = load_json_file(metadata_path / "tracking_metadata.json") or {}
        alerts = load_json_file(metadata_path / "alert.json") or {"alerts": []}
        session_metadata = load_json_file(metadata_path / "session_metadata.json") or {}
        
        response = ""
        
        if "alert" in search_text or "critical" in search_text:
            alert_list = alerts.get("alerts", [])
            critical = [a for a in alert_list if a.get("severity") == "Critical"]
            response = f"Found {len(alert_list)} total alerts, {len(critical)} critical."
            
        elif "track" in search_text and any(c.isdigit() for c in search_text):
            import re
            numbers = re.findall(r'\d+', search_text)
            if numbers:
                threshold = int(numbers[0])
                tracks = tracking_metadata.get("tracks", {})
                track_lengths = Counter()
                for frame_tracks in tracks.values():
                    for track in frame_tracks:
                        track_lengths[track["track_id"]] += 1
                
                fps = session_metadata.get("fps", 30)
                long_tracks = [t for t, frames in track_lengths.items() if frames / fps > threshold]
                response = f"Found {len(long_tracks)} tracks longer than {threshold} seconds."
                
        elif any(cls in search_text for cls in ["person", "vehicle", "weapon", "backpack"]):
            detections = detection_metadata.get("detections", {})
            class_counts = Counter()
            for frame_dets in detections.values():
                for det in frame_dets:
                    class_counts[det.get("cls", "unknown")] += 1
            
            for cls in ["person", "vehicle", "weapon", "backpack"]:
                if cls in search_text:
                    count = class_counts.get(cls, 0)
                    response = f"Detected {count} {cls} instances in this session."
                    break
                    
        elif "summary" in search_text or "overview" in search_text:
            report = load_json_file(metadata_path / "report.json") or {}
            summary = report.get("summary_for_user", {})
            risk = summary.get("overall_risk_level", "Unknown")
            response = f"Session {session_id}\nRisk Level: {risk}\nAlerts: {len(alerts.get('alerts', []))}"
        
        else:
            response = "Query not understood. Try: 'show alerts', 'tracks longer than 30s', 'person detections', 'summary'"
        
        return {"response": response}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)