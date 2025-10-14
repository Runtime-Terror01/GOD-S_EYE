## Storage layout (must be exactly followed)
```
surveillance_storage/
  session_<session_id>/
    raw_frame/
      frame_000000.jpg, frame_000001.jpg, ...
    annotated_frame/
      frame_000000.jpg, ...
    metadata/
      session_metadata.json
      frames_metadata.json      # array of per-frame objects
      detection_metadata.json
      tracking_metadata.json
      alert.json
      report.json
    vectors/                    # for vector DB payload(s) (jsonl or embedding files)
```

# Standardized metadata formats

## 1. session_metadata.json:
```
{
  "session_id": "ertyu",
  "video_path": "input/my_video.mp4",
  "started_at": "2025-10-14T09:00:00+05:30",
  "frame_count": 1200,
  "fps": 30
}
```
## 2. frames_metadata.json:
```
[
  {
    "session_id": "ertyu",
    "frame_id": "f01fb80b-0e78-44ab-bfb4-69a26da19ecd",
    "frame_index": 0,
    "timestamp_sec": 0.0,
    "path": "surveillance_storage/session_ertyu/raw_frame/frame_000000.jpg"
  },
  ...
]
```
## 3. detection_metadata.json:
```
{
  "session_id": "ertyu",
  "detections": {
    "frame_000000": [
      {"cls": "person", "bbox": [100,150,230,400], "conf": 0.92},
      {"cls": "car", "bbox": [300,200,500,450], "conf": 0.87}
    ],
    ...
  }
}
```
## 4. tracking_metadata.json:
```
{
  "session_id": "ertyu",
  "tracks": {
    "frame_000000": [
      {"track_id": 1, "cls": "person", "bbox": [100,150,230,400], "conf": 0.92},
      {"track_id": 2, "cls": "car", "bbox": [300,200,500,450], "conf": 0.87}
    ],
    ...
  }
}
```
## 5. alert.json
```
{
  "session_id": "ertyu",
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
## 6. report.json
```
{
  "session_id": "ertyu",
  "comprehensive_report": "### COMPREHENSIVE REPORT ###\nTIMESTAMP: 00:00:05 - 00:00:08\nOBSERVATIONS: ...\nFLAGGED_ITEMS: ...\n---\n",
  "summary_for_user": {
    "overall_risk_level": "Medium",
    "justification": "Short paragraph...",
    "recommended_actions": ["Action 1", "Action 2"]
  },
  "generated_at": "2025-10-14T09:30:00+05:30"
}
```
## vlm_alert.json
```
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


## Dashboard backend (Flask / FastAPI) — minimal spec

Use FastAPI (preferred) with Uvicorn for dev.

Config: base SURVEILLANCE_ROOT path points to surveillance_storage.

Endpoints:

GET /sessions → list session ids and session_metadata

GET /sessions/{session_id}/frames?start=&end=&limit= → paginated frames metadata

GET /sessions/{session_id}/frames/{frame_id} → returns frame image (serve file)

GET /sessions/{session_id}/detections → detection_metadata.json

GET /sessions/{session_id}/tracks → tracking_metadata.json

GET /sessions/{session_id}/alerts → alert.json

GET /sessions/{session_id}/report → report.json

POST /search → semantic search wrapper that queries vector files or a local mock vector DB (return matching session/report snippets)

The backend must watch for new files in surveillance_storage and refresh caches (polling or simple in-memory TTL cache). Simple polling with watchdog is acceptable but optional — a periodic refresh every X seconds is enough for prototype.

Return JSON only. Keep CORS enabled for frontend dev.


## Dashboard frontend hooks (minimum)

Provide a minimal React app skeleton or HTML page with:

Session list (fetch /sessions)

Frame viewer (fetch /sessions/{id}/frames/{frame_id})

Alerts panel (poll /sessions/{id}/alerts)

Chatbox that posts queries to /search and displays VLM summaries

You can keep this part extremely minimal — focus on clear API usage.