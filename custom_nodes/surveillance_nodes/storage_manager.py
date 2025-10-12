"""
Robust Storage Management System for Surveillance Pipeline
Handles frames, metadata, and embeddings with proper referencing
"""

import os
import json
import uuid
import hashlib
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
import queue
from dataclasses import asdict
import pickle

# Database connectors (examples - replace with actual implementations)
try:
    from pymongo import MongoClient
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Install pymongo and chromadb for full functionality")
    
config = {
    'storage_root': './surveillance_storage',
    'mongo_uri': 'mongodb://localhost:27017/',
    'chroma_path': './surveillance_storage/chroma'
}


class StorageManager:
    """Central storage management system for all pipeline data"""
    
    def __init__(self, config: Dict[str, str] = None):
        """
        Initialize storage manager with configuration
        
        Args:
            config: Dictionary containing:
                - storage_root: Root directory for file storage
                - mongo_uri: MongoDB connection string
                - chroma_path: ChromaDB storage path
        """
        self.config = config or {}
        
        # File storage setup
        self.storage_root = Path(self.config.get('storage_root', './surveillance_storage'))
        self.frames_dir = self.storage_root / 'frames'
        self.temp_dir = self.storage_root / 'temp'
        self.archive_dir = self.storage_root / 'archive'
        
        # Create directories
        for directory in [self.frames_dir, self.temp_dir, self.archive_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        self._init_nosql_db()
        self._init_vector_db()
        
        # Session management
        self.active_sessions = {}
        self.session_lock = threading.Lock()
        
        # Write queue for async operations
        self.write_queue = queue.Queue()
        self.write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.write_thread.start()
    
    def _init_nosql_db(self):
        """Initialize NoSQL database connection"""
        try:
            mongo_uri = self.config.get('mongo_uri', 'mongodb://localhost:27017/')
            self.mongo_client = MongoClient(mongo_uri)
            self.metadata_db = self.mongo_client['surveillance_metadata']
            
            # Create collections with indexes
            self._create_collections_with_indexes()
            print("✓ Connected to MongoDB")
        except Exception as e:
            print(f"Warning: MongoDB connection failed: {e}")
            self.metadata_db = None
    
    def _init_vector_db(self):
        """Initialize vector database for embeddings"""
        try:
            chroma_path = self.config.get('chroma_path', str(self.storage_root / 'chroma'))
            self.chroma_client = chromadb.PersistentClient(
                path=chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create collections for different embedding types
            self.scene_collection = self.chroma_client.get_or_create_collection(
                name="scene_embeddings",
                metadata={"description": "Scene understanding embeddings"}
            )
            
            self.face_collection = self.chroma_client.get_or_create_collection(
                name="face_embeddings",
                metadata={"description": "Face recognition embeddings"}
            )
            print("✓ Connected to ChromaDB")
        except Exception as e:
            print(f"Warning: ChromaDB initialization failed: {e}")
            self.chroma_client = None
    
    def _create_collections_with_indexes(self):
        """Create MongoDB collections with appropriate indexes"""
        if not self.metadata_db:
            return
        
        collections = {
            'sessions': ['session_id', 'timestamp', 'pipeline_id'],
            'frames': ['frame_id', 'session_id', 'timestamp', 'frame_index'],
            'detections': ['detection_id', 'frame_id', 'class_name', 'timestamp'],
            'tracks': ['track_id', 'session_id', 'start_time', 'end_time'],
            'alerts': ['alert_id', 'session_id', 'timestamp', 'threat_level'],
            'vlm_reports': ['report_id', 'session_id', 'timestamp'],
            'audit_log': ['timestamp', 'session_id', 'action']
        }
        
        for collection_name, indexes in collections.items():
            collection = self.metadata_db[collection_name]
            for index in indexes:
                collection.create_index(index)
    
    # ==================== Session Management ====================
    
    def create_session(self, pipeline_id: str, source_video: str) -> str:
        """
        Create a new processing session
        
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        
        session_data = {
            'session_id': session_id,
            'pipeline_id': pipeline_id,
            'source_video': source_video,
            'timestamp': datetime.now(),
            'status': 'active',
            'storage_path': str(self.frames_dir / session_id)
        }
        
        # Create session-specific storage directory
        session_dir = self.frames_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Store session in database
        if self.metadata_db:
            self.metadata_db.sessions.insert_one(session_data)
        
        # Track active session
        with self.session_lock:
            self.active_sessions[session_id] = session_data
        
        self._log_audit("session_created", session_id, {"pipeline_id": pipeline_id})
        
        return session_id
    
    def close_session(self, session_id: str):
        """Close and archive a session"""
        with self.session_lock:
            if session_id in self.active_sessions:
                session_data = self.active_sessions.pop(session_id)
                
                # Update session status in database
                if self.metadata_db:
                    self.metadata_db.sessions.update_one(
                        {'session_id': session_id},
                        {'$set': {'status': 'completed', 'end_time': datetime.now()}}
                    )
                
                self._log_audit("session_closed", session_id, {})
    
    # ==================== Frame Storage ====================
    
    def store_frame(self, frame_data: 'FrameData', session_id: str) -> Dict[str, str]:
        """
        Store frame data with all references
        
        Returns:
            Dictionary with storage paths and IDs
        """
        # Validate session
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")
        
        # Generate unique frame ID if not provided
        if not frame_data.frame_id:
            frame_data.frame_id = str(uuid.uuid4())
        
        # Prepare frame metadata
        frame_metadata = {
            'frame_id': frame_data.frame_id,
            'session_id': session_id,
            'frame_index': frame_data.frame_index,
            'timestamp': frame_data.timestamp,
            'storage_path': frame_data.path,
            'width': frame_data.width,
            'height': frame_data.height,
            'metadata': frame_data.metadata or {},
            'created_at': datetime.now()
        }
        
        # Store in database
        if self.metadata_db:
            self.metadata_db.frames.insert_one(frame_metadata)
        
        # Create reference mapping
        reference = {
            'frame_id': frame_data.frame_id,
            'session_id': session_id,
            'storage_path': frame_data.path,
            'db_collection': 'frames'
        }
        
        return reference
    
    def store_frames_batch(self, frames: List['FrameData'], session_id: str):
        """Store multiple frames efficiently"""
        references = []
        frame_documents = []
        
        for frame in frames:
            if not frame.frame_id:
                frame.frame_id = str(uuid.uuid4())
            
            frame_doc = {
                'frame_id': frame.frame_id,
                'session_id': session_id,
                'frame_index': frame.frame_index,
                'timestamp': frame.timestamp,
                'storage_path': frame.path,
                'width': frame.width,
                'height': frame.height,
                'metadata': frame.metadata or {},
                'created_at': datetime.now()
            }
            frame_documents.append(frame_doc)
            
            references.append({
                'frame_id': frame.frame_id,
                'session_id': session_id,
                'storage_path': frame.path
            })
        
        # Batch insert for efficiency
        if self.metadata_db and frame_documents:
            self.metadata_db.frames.insert_many(frame_documents)
        
        return references
    
    # ==================== Detection & Track Storage ====================
    
    def store_detection(self, detection: 'Detection', session_id: str) -> str:
        """Store detection data with references to frames"""
        detection_doc = {
            'detection_id': detection.detection_id,
            'session_id': session_id,
            'frame_id': detection.frame_id,
            'class_name': detection.class_name,
            'class_id': detection.class_id,
            'bbox': asdict(detection.bbox),
            'confidence': detection.confidence,
            'attributes': detection.attributes or {},
            'timestamp': detection.timestamp,
            'created_at': datetime.now()
        }
        
        if self.metadata_db:
            self.metadata_db.detections.insert_one(detection_doc)
        
        return detection.detection_id
    
    def store_track(self, track: 'Track', session_id: str) -> int:
        """Store tracking data with all detections"""
        track_doc = {
            'track_id': track.track_id,
            'session_id': session_id,
            'detection_ids': [d.detection_id for d in track.detections],
            'start_time': track.start_time,
            'end_time': track.end_time,
            'duration': track.duration,
            'trajectory': track.trajectory,
            'attributes': track.attributes or {},
            'created_at': datetime.now()
        }
        
        if self.metadata_db:
            # Store track
            self.metadata_db.tracks.insert_one(track_doc)
            
            # Store associated detections
            for detection in track.detections:
                self.store_detection(detection, session_id)
        
        return track.track_id
    
    # ==================== Alert Storage ====================
    
    def store_alert(self, alert: 'Alert', session_id: str) -> str:
        """Store alert with all evidence references"""
        alert_doc = {
            'alert_id': alert.alert_id,
            'session_id': session_id,
            'alert_type': alert.alert_type,
            'threat_level': alert.threat_level.value,
            'timestamp': alert.timestamp,
            'location': alert.location,
            'description': alert.description,
            'evidence': alert.evidence or {},
            'frame_ids': alert.frame_ids or [],
            'track_ids': alert.track_ids or [],
            'requires_action': alert.requires_action,
            'assigned_to': alert.assigned_to,
            'status': 'open',
            'created_at': datetime.now()
        }
        
        if self.metadata_db:
            self.metadata_db.alerts.insert_one(alert_doc)
        
        # Send to dashboard via message queue (implement based on your setup)
        self._send_alert_to_dashboard(alert_doc)
        
        return alert.alert_id
    
    # ==================== VLM Report Storage ====================
    
    def store_vlm_report(self, report: 'VLMReport', embeddings: List[float] = None) -> str:
        """Store VLM report with optional embeddings for semantic search"""
        report_doc = {
            'report_id': report.report_id,
            'session_id': report.session_id,
            'timestamp': report.timestamp,
            'comprehensive_report': report.comprehensive_report,
            'executive_summary': report.executive_summary,
            'risk_assessment': report.risk_assessment.value,
            'key_events': report.key_events,
            'recommendations': report.recommendations,
            'created_at': datetime.now()
        }
        
        # Store in NoSQL database
        if self.metadata_db:
            self.metadata_db.vlm_reports.insert_one(report_doc)
        
        # Store embeddings for semantic search
        if embeddings and self.scene_collection:
            # Split report into chunks for better search
            chunks = self._split_report_into_chunks(report.comprehensive_report)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{report.report_id}_chunk_{i}"
                
                self.scene_collection.add(
                    ids=[chunk_id],
                    embeddings=[embeddings[i] if i < len(embeddings) else embeddings[-1]],
                    metadatas=[{
                        'report_id': report.report_id,
                        'session_id': report.session_id,
                        'chunk_index': i,
                        'timestamp': report.timestamp.isoformat(),
                        'risk_level': report.risk_assessment.value
                    }],
                    documents=[chunk]
                )
        
        return report.report_id
    
    # ==================== Query Methods ====================
    
    def get_frame_by_id(self, frame_id: str) -> Optional[Dict]:
        """Retrieve frame metadata by ID"""
        if self.metadata_db:
            return self.metadata_db.frames.find_one({'frame_id': frame_id})
        return None
    
    def get_frames_by_session(self, session_id: str, 
                             start_time: float = None, 
                             end_time: float = None) -> List[Dict]:
        """Retrieve frames for a session within time range"""
        query = {'session_id': session_id}
        
        if start_time is not None or end_time is not None:
            query['timestamp'] = {}
            if start_time is not None:
                query['timestamp']['$gte'] = start_time
            if end_time is not None:
                query['timestamp']['$lte'] = end_time
        
        if self.metadata_db:
            return list(self.metadata_db.frames.find(query).sort('timestamp', 1))
        return []
    
    def semantic_search(self, query_text: str, 
                       query_embedding: List[float],
                       n_results: int = 5) -> List[Dict]:
        """Perform semantic search on VLM reports"""
        if not self.scene_collection:
            return []
        
        results = self.scene_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['metadatas', 'documents', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted_results
    
    def get_alerts_by_threat_level(self, threat_level: str, 
                                   session_id: str = None) -> List[Dict]:
        """Get alerts filtered by threat level"""
        query = {'threat_level': threat_level}
        if session_id:
            query['session_id'] = session_id
        
        if self.metadata_db:
            return list(self.metadata_db.alerts.find(query).sort('timestamp', -1))
        return []
    
    # ==================== Utility Methods ====================
    
    def _split_report_into_chunks(self, report_text: str, 
                                  chunk_size: int = 500) -> List[str]:
        """Split long report into chunks for better embedding search"""
        words = report_text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _send_alert_to_dashboard(self, alert_doc: Dict):
        """Send alert to dashboard server"""
        # Implement based on your dashboard communication method
        # Example: Redis pub/sub, WebSocket, REST API
        pass
    
    def _write_worker(self):
        """Background worker for async write operations"""
        while True:
            try:
                task = self.write_queue.get(timeout=1)
                if task is None:
                    break
                
                task_type = task.get('type')
                if task_type == 'frame':
                    self.store_frame(task['data'], task['session_id'])
                elif task_type == 'detection':
                    self.store_detection(task['data'], task['session_id'])
                # Add more task types as needed
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Write worker error: {e}")
    
    def _log_audit(self, action: str, session_id: str, details: Dict):
        """Log audit trail for compliance"""
        audit_entry = {
            'timestamp': datetime.now(),
            'session_id': session_id,
            'action': action,
            'details': details
        }
        
        if self.metadata_db:
            self.metadata_db.audit_log.insert_one(audit_entry)
    
    def cleanup_old_sessions(self, days: int = 30):
        """Clean up old sessions and their data"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if self.metadata_db:
            # Find old sessions
            old_sessions = self.metadata_db.sessions.find({
                'timestamp': {'$lt': cutoff_date},
                'status': 'completed'
            })
            
            for session in old_sessions:
                session_id = session['session_id']
                
                # Archive or delete frame files
                session_dir = self.frames_dir / session_id
                if session_dir.exists():
                    archive_path = self.archive_dir / f"{session_id}.tar.gz"
                    # Create archive (implement compression)
                    shutil.rmtree(session_dir)
                
                # Clean up database entries
                for collection in ['frames', 'detections', 'tracks', 'alerts']:
                    self.metadata_db[collection].delete_many({'session_id': session_id})
                
                self._log_audit("session_archived", session_id, {'days_old': days})
    
    def get_storage_stats(self) -> Dict:
        """Get storage usage statistics"""
        stats = {
            'total_sessions': len(self.active_sessions),
            'storage_used_gb': sum(f.stat().st_size for f in self.storage_root.rglob('*') if f.is_file()) / (1024**3),
            'database_stats': {}
        }
        
        if self.metadata_db:
            for collection_name in ['sessions', 'frames', 'detections', 'tracks', 'alerts']:
                stats['database_stats'][collection_name] = self.metadata_db[collection_name].count_documents({})
        
        return stats

from datetime import timedelta