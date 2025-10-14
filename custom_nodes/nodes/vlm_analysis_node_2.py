"""
VLM Analysis Node with Vector Database Integration
Analyzes full video using Gemini API and stores reports in vector database for semantic search
"""

import os
import time
import uuid
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timezone
import hashlib

import google.generativeai as genai
import numpy as np

# Vector database imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not available. Install with: pip install chromadb")

# Alternative: Use FAISS if ChromaDB not available
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu sentence-transformers")

from .base_node import BaseNode
from .data_types import VIDEO, REPORT, ALERTS

# --- AI Configuration ---
MODEL_NAME = "gemini-2.0-flash-exp"  # or "gemini-1.5-pro" based on availability

SYSTEM_INSTRUCTION = (
    "You are a highly specialized Security Footage Analysis Agent. "
    "Your task is to analyze video footage and generate a two-part security report. "
    "Your response MUST be objective, factual, and strictly adhere to the requested format. "
    "Do not add any conversational text outside of the requested structure."
)

USER_QUERY = (
    "Analyze the provided video from start to finish and generate a security report in two distinct sections. "
    "**It is critical that you do not stop prematurely and analyze the entire duration of the video until the final second.**\n\n"
    "Section 1: A comprehensive, machine-readable event log suitable for a database. Use the following simplified format for each notable event or time segment:\n"
    "TIMESTAMP: [HH:MM:SS] - [HH:MM:SS]\n"
    "OBSERVATIONS: [Detailed, objective description of all events, subjects, and actions.]\n"
    "FLAGGED_ITEMS: [List any suspicious behavior, unauthorized access, weapons, or harmful objects. State 'None' if nothing is found.]\n"
    "--- (use this separator between entries) ---\n\n"
    "Section 2: An actionable, human-readable summary for a security operator. It MUST follow this exact three-part format:\n"
    "OVERALL RISK LEVEL: [None, Low, Medium, High, Critical]\n"
    "JUSTIFICATION: [A concise paragraph explaining the key events that led to the assigned risk level.]\n"
    "RECOMMENDED ACTIONS: [A bulleted list of concrete, actionable steps for a security team to take. Examples: 'Immediately escalate to law enforcement due to credible threat.', 'Review access logs for the time of the forced entry.', 'Archive for informational purposes; no immediate action required.']\n\n"
    "Begin your entire response with '### COMPREHENSIVE REPORT ###' and '### SUMMARY FOR USER ###' to delimit the sections."
)


class VectorDatabase:
    """Abstract interface for vector database operations"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def add_document(self, session_id: str, document: str, metadata: Dict[str, Any]):
        raise NotImplementedError
        
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError


class ChromaDBVectorStore(VectorDatabase):
    """ChromaDB implementation for vector storage"""
    
    def __init__(self, storage_path: str):
        super().__init__(storage_path)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.storage_path / "chromadb"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection for surveillance reports
        self.collection = self.client.get_or_create_collection(
            name="surveillance_reports",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, session_id: str, document: str, metadata: Dict[str, Any]):
        """Add document to ChromaDB"""
        # Split document into chunks for better search
        chunks = self._split_into_chunks(document)
        
        for i, chunk in enumerate(chunks):
            doc_id = f"{session_id}_chunk_{i}"
            
            self.collection.add(
                documents=[chunk],
                metadatas=[{
                    "session_id": session_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **metadata
                }],
                ids=[doc_id]
            )
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def _split_into_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into chunks for better retrieval"""
        # Split by event separator first
        events = text.split('---')
        chunks = []
        
        for event in events:
            event = event.strip()
            if not event:
                continue
                
            # If event is too long, split further
            if len(event) > max_chunk_size:
                # Split by sentences
                sentences = event.replace('. ', '.|').split('|')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk)
            else:
                chunks.append(event)
        
        return chunks


class FAISSVectorStore(VectorDatabase):
    """FAISS implementation for vector storage (fallback)"""
    
    def __init__(self, storage_path: str):
        super().__init__(storage_path)
        
        # Initialize sentence transformer for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load or create index
        self.index_path = self.storage_path / "faiss_index"
        self.metadata_path = self.storage_path / "faiss_metadata.json"
        
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = self._load_metadata()
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
    
    def add_document(self, session_id: str, document: str, metadata: Dict[str, Any]):
        """Add document to FAISS index"""
        chunks = self._split_into_chunks(document)
        
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.model.encode([chunk])
            
            # Add to index
            self.index.add(embedding)
            
            # Store metadata
            self.metadata.append({
                "session_id": session_id,
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **metadata
            })
        
        # Save index and metadata
        self._save()
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding, min(limit, self.index.ntotal))
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    'text': self.metadata[idx]['text'],
                    'metadata': self.metadata[idx],
                    'distance': float(dist)
                })
        
        return results
    
    def _split_into_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into chunks"""
        events = text.split('---')
        chunks = []
        
        for event in events:
            event = event.strip()
            if not event:
                continue
                
            if len(event) > max_chunk_size:
                # Simple chunking by character count
                for i in range(0, len(event), max_chunk_size):
                    chunks.append(event[i:i+max_chunk_size])
            else:
                chunks.append(event)
        
        return chunks
    
    def _save(self):
        """Save index and metadata to disk"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load metadata from disk"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return []


class VLMAnalysisNode(BaseNode):
    """
    VLM Analysis Node using Gemini API with Vector Database Integration
    Analyzes full video and generates comprehensive security reports
    """
    
    # ComfyUI Node Configuration
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter Gemini API key"
                }),
                "execute": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_query": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional custom analysis query"
                }),
                "custom_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional custom system instruction"
                }),
                "storage_path": ("STRING", {
                    "default": "surveillance_storage",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("REPORT", "ALERTS")
    RETURN_NAMES = ("report", "alerts")
    FUNCTION = "analyze"
    CATEGORY = "surveillance"
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.vector_db = None
        self.session_id = None
        self.api_key = None
    
    def analyze(self, video, api_key: str, execute: bool, 
                custom_query: str = "", custom_instruction: str = "", 
                storage_path: str = "surveillance_storage"):
        """ComfyUI entry point"""
        if not execute:
            return (None, [])
        
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        self.api_key = api_key
        return self.process(video, storage_path, custom_query, custom_instruction)
    
    def process(self, video: VIDEO, storage_path: str, 
                custom_query: str = "", custom_instruction: str = "") -> Tuple[REPORT, ALERTS]:
        """
        Process video through VLM and generate security report
        
        Args:
            video: Video input (path or VIDEO object)
            storage_path: Base storage directory
            custom_query: Optional custom analysis query
            custom_instruction: Optional custom system instruction
            
        Returns:
            (report, alerts): Report dictionary and list of alerts
        """
        # Initialize Gemini API
        genai.configure(api_key=self.api_key)
        
        # Generate session ID
        self.session_id = str(uuid.uuid4())[:8]
        
        # Get video path
        video_path = self._get_video_path(video)
        
        print(f"[VLM] Processing video: {video_path}")
        print(f"[VLM] Session ID: {self.session_id}")
        
        # Analyze video with Gemini
        comprehensive_report, summary = self._analyze_video_with_gemini(
            video_path,
            custom_query or USER_QUERY,
            custom_instruction or SYSTEM_INSTRUCTION
        )
        
        # Parse summary for structured data
        risk_level, justification, actions = self._parse_summary(summary)
        
        # Create report structure
        report = {
            "session_id": self.session_id,
            "comprehensive_report": comprehensive_report,
            "summary_for_user": {
                "overall_risk_level": risk_level,
                "justification": justification,
                "recommended_actions": actions
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "video_path": video_path
        }
        
        # Generate alerts from report
        alerts = self._generate_alerts_from_report(comprehensive_report, risk_level)
        
        # Save metadata
        session_dir = Path(storage_path) / f"session_{self.session_id}"
        metadata_dir = session_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_path = metadata_dir / "report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save alerts
        if alerts:
            alerts_path = metadata_dir / "alerts.json"
            with open(alerts_path, 'w') as f:
                json.dump({"session_id": self.session_id, "alerts": alerts}, f, indent=2)
        
        print(f"[VLM] ✅ Report saved to {report_path}")
        
        # Initialize and update vector database
        self._initialize_vector_db(storage_path)
        self._add_to_vector_db(report)
        
        print(f"[VLM] ✅ Added to vector database for semantic search")
        
        return report, alerts
    
    # M:\Order_to_PC\MVP\GOD'S_EYE\custom_nodes\nodes\vlm_analysis_node.py

    def _get_video_path(self, video_input) -> str:
        """Extract video file path from various input types"""
        # --- Optional: Uncomment these lines for debugging if the error persists ---
        # print(f"DEBUG: Inspecting video_input. Type: {type(video_input)}")
        # try:
        #     print(f"DEBUG: Attributes (dir): {dir(video_input)}")
        #     print(f"DEBUG: Dictionary view (vars): {vars(video_input)}")
        # except TypeError:
        #     pass # vars() might fail on some types
        # --------------------------------------------------------------------

        # 1. Check if it's already a path string
        if isinstance(video_input, (str, Path)):
            return str(video_input)
        
        # 2. Check for common direct attributes
        for attr in ['filename', 'path', 'file_path', 'filepath', 'source', 'video_path']:
            if hasattr(video_input, attr):
                value = getattr(video_input, attr)
                if isinstance(value, (str, Path)):
                    return str(value)

        # 3. Check for dictionary-like access (Most likely solution)
        if hasattr(video_input, 'get'):
            # ADDED 'filename' to this list of keys
            for key in ['filename', 'path', 'file_path', 'video_path', 'source']:
                value = video_input.get(key)
                if value and isinstance(value, (str, Path)):
                    return str(value)
        
        # If all else fails, raise the error
        raise ValueError(f"Could not extract video path from input type: {type(video_input)}")
        
        # Try dictionary-like access
        if hasattr(video_input, 'get'):
            for key in ['path', 'file_path', 'video_path', 'source']:
                value = video_input.get(key)
                if value and isinstance(value, (str, Path)):
                    return str(value)
        
        raise ValueError(f"Could not extract video path from input type: {type(video_input)}")
    
    def _analyze_video_with_gemini(self, video_path: str, query: str, instruction: str) -> Tuple[str, str]:
        """
        Analyze video using Gemini API
        
        Returns:
            (comprehensive_report, summary): Parsed sections from Gemini response
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print("[VLM] Uploading video to Gemini...")
        
        # Upload video
        uploaded_file = genai.upload_file(path=video_path)
        print(f"[VLM] ✅ Uploaded: {uploaded_file.display_name}")
        
        # Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            print("[VLM] ⏳ Processing video...")
            time.sleep(5)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            raise RuntimeError("Video processing failed")
        
        # Generate response
        print("[VLM] Generating analysis...")
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=instruction
        )
        
        response = model.generate_content([
            uploaded_file,
            query
        ])
        
        # Parse response
        parsed = self._parse_model_response(response.text)
        
        # Cleanup
        try:
            genai.delete_file(uploaded_file.name)
        except:
            pass
        
        if not parsed:
            raise ValueError("Failed to parse Gemini response")
        
        return parsed[0], parsed[1]
    
    def _parse_model_response(self, response_text: str) -> Optional[List[str]]:
        """Parse Gemini response into comprehensive report and summary"""
        try:
            report_marker = "### COMPREHENSIVE REPORT ###"
            summary_marker = "### SUMMARY FOR USER ###"
            
            report_start = response_text.find(report_marker)
            summary_start = response_text.find(summary_marker)
            
            if report_start == -1 or summary_start == -1:
                return None
            
            report_content = response_text[report_start + len(report_marker):summary_start].strip()
            summary_content = response_text[summary_start + len(summary_marker):].strip()
            
            if not report_content or not summary_content:
                return None
            
            return [report_content, summary_content]
        except Exception:
            return None
    
    def _parse_summary(self, summary: str) -> Tuple[str, str, List[str]]:
        """Parse summary section for structured data"""
        risk_level = "Unknown"
        justification = ""
        actions = []
        
        lines = summary.split('\n')
        
        for i, line in enumerate(lines):
            if line.startswith("OVERALL RISK LEVEL:"):
                risk_level = line.split(":", 1)[1].strip()
            elif line.startswith("JUSTIFICATION:"):
                # Get justification paragraph
                j = i + 1
                justification_lines = []
                while j < len(lines) and not lines[j].startswith("RECOMMENDED ACTIONS:"):
                    justification_lines.append(lines[j])
                    j += 1
                justification = " ".join(justification_lines).strip()
            elif line.startswith("RECOMMENDED ACTIONS:"):
                # Get actions list
                j = i + 1
                while j < len(lines):
                    action = lines[j].strip()
                    if action.startswith("- ") or action.startswith("* "):
                        actions.append(action[2:].strip())
                    elif action:
                        actions.append(action)
                    j += 1
        
        return risk_level, justification, actions
    
    def _generate_alerts_from_report(self, report: str, risk_level: str) -> List[Dict[str, Any]]:
        """Generate alerts based on report content"""
        alerts = []
        alert_id = 0
        
        # Parse events from comprehensive report
        events = report.split('---')
        
        for event in events:
            event = event.strip()
            if not event:
                continue
            
            # Extract timestamp and flagged items
            timestamp = None
            flagged_items = None
            
            lines = event.split('\n')
            for line in lines:
                if line.startswith("TIMESTAMP:"):
                    timestamp = line.split(":", 1)[1].strip()
                elif line.startswith("FLAGGED_ITEMS:"):
                    flagged_items = line.split(":", 1)[1].strip()
            
            # Create alert if there are flagged items
            if flagged_items and flagged_items.lower() != "none":
                # Convert timestamp to seconds (simplified)
                timestamp_sec = self._timestamp_to_seconds(timestamp)
                
                # Determine severity based on content
                severity = self._determine_severity(flagged_items, risk_level)
                
                alerts.append({
                    "alert_id": f"alert_{alert_id:04d}",
                    "timestamp_sec": timestamp_sec,
                    "frame_ref": f"time_{timestamp}",
                    "reason": "suspicious_activity",
                    "evidence": {
                        "flagged_items": flagged_items,
                        "event_description": event[:200]  # First 200 chars
                    },
                    "severity": severity
                })
                alert_id += 1
        
        return alerts
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp string to seconds"""
        try:
            # Handle range timestamps like "00:00:05 - 00:00:08"
            if " - " in timestamp:
                start_time = timestamp.split(" - ")[0]
            else:
                start_time = timestamp
            
            # Parse HH:MM:SS
            parts = start_time.split(":")
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        except:
            pass
        
        return 0.0
    
    def _determine_severity(self, flagged_items: str, risk_level: str) -> str:
        """Determine alert severity"""
        flagged_lower = flagged_items.lower()
        
        # Check for critical keywords
        if any(word in flagged_lower for word in ['weapon', 'gun', 'knife', 'explosive', 'bomb']):
            return "Critical"
        elif any(word in flagged_lower for word in ['unauthorized', 'breach', 'intrusion', 'forced']):
            return "High"
        elif any(word in flagged_lower for word in ['suspicious', 'unusual', 'loitering']):
            return "Medium"
        else:
            # Use overall risk level as fallback
            if risk_level in ["Critical", "High", "Medium", "Low"]:
                return risk_level
            return "Low"
    
    def _initialize_vector_db(self, storage_path: str):
        """Initialize vector database"""
        if self.vector_db is not None:
            return
        
        vector_db_path = Path(storage_path) / "vectors"
        
        if CHROMADB_AVAILABLE:
            self.vector_db = ChromaDBVectorStore(str(vector_db_path))
            print("[VLM] Using ChromaDB for vector storage")
        elif FAISS_AVAILABLE:
            self.vector_db = FAISSVectorStore(str(vector_db_path))
            print("[VLM] Using FAISS for vector storage")
        else:
            print("[VLM] Warning: No vector database available. Semantic search disabled.")
            self.vector_db = None
    
    def _add_to_vector_db(self, report: Dict[str, Any]):
        """Add report to vector database for semantic search"""
        if self.vector_db is None:
            return
        
        # Prepare document for indexing
        document = f"{report['comprehensive_report']}\n\n{report['summary_for_user']['justification']}"
        
        # Prepare metadata
        metadata = {
            "risk_level": report['summary_for_user']['overall_risk_level'],
            "generated_at": report['generated_at'],
            "video_path": report.get('video_path', '')
        }
        
        # Add to vector database
        self.vector_db.add_document(report['session_id'], document, metadata)


class SemanticSearchNode(BaseNode):
    """
    Node for performing semantic search on VLM reports
    Can be used in ComfyUI workflows or via API
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter search query"
                }),
                "storage_path": ("STRING", {
                    "default": "surveillance_storage",
                    "multiline": False
                }),
                "max_results": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("search_results",)
    FUNCTION = "search"
    CATEGORY = "surveillance"
    
    def search(self, query: str, storage_path: str, max_results: int = 5):
        """Perform semantic search"""
        if not query:
            return ("No query provided",)
        
        vector_db_path = Path(storage_path) / "vectors"
        
        # Initialize appropriate vector database
        if CHROMADB_AVAILABLE:
            vector_db = ChromaDBVectorStore(str(vector_db_path))
        elif FAISS_AVAILABLE:
            vector_db = FAISSVectorStore(str(vector_db_path))
        else:
            return ("Vector database not available. Install ChromaDB or FAISS.",)
        
        # Perform search
        results = vector_db.search(query, limit=max_results)
        
        # Format results
        if not results:
            return ("No results found",)
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"Result {i}:")
            formatted.append(f"Session: {result['metadata'].get('session_id', 'Unknown')}")
            formatted.append(f"Risk Level: {result['metadata'].get('risk_level', 'Unknown')}")
            formatted.append(f"Text: {result['text'][:200]}...")
            formatted.append("---")
        
        return ("\n".join(formatted),)