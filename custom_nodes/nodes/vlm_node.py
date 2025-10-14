"""
VLM Node - Video Language Model for scene understanding and report generation
Uses Google Gemini API to analyze video and generate comprehensive reports
"""

import os
import time
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime, timezone

from .base_node import BaseNode
from .data_types import VIDEO, REPORT

# Try to import Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not available. Using mock VLM.")


class VLMNode(BaseNode):
    """
    Video Language Model node for scene understanding
    Uses LLM (Google Gemini) to analyze video and generate reports
    Outputs REPORT dict
    Saves report.json and vectors
    """
    
    DEFAULT_SYSTEM_INSTRUCTION = (
        "You are a highly specialized Security Footage Analysis Agent. "
        "Your task is to analyze video footage and generate a two-part security report. "
        "Your response MUST be objective, factual, and strictly adhere to the requested format. "
        "Do not add any conversational text outside of the requested structure."
    )
    
    DEFAULT_USER_QUERY = (
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
    
    # ComfyUI Node Configuration
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "input/video.mp4"}),
            },
            "optional": {
                "session_id": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("REPORT",)
    RETURN_NAMES = ("report",)
    FUNCTION = "analyze"
    CATEGORY = "surveillance"
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = None
        self.model_name = "gemini-2.0-flash-exp"
        self.session_id = None
    
    def analyze(self, video_path: str, session_id: str = ""):
        """ComfyUI entry point"""
        if session_id:
            self.config['session_id'] = session_id
        return self.process(video_path)
        
    def process(self, video_path: str, system_instruction: str = None, user_query: str = None) -> Tuple[REPORT]:
        """
        Analyze video using VLM and generate report
        
        Args:
            video_path: Path to video file
            system_instruction: System instruction for LLM (optional)
            user_query: User query for analysis (optional)
            
        Returns:
            report: Dictionary matching report.json structure
        """
        # Use default prompts if not provided
        if system_instruction is None:
            system_instruction = self.DEFAULT_SYSTEM_INSTRUCTION
        if user_query is None:
            user_query = self.DEFAULT_USER_QUERY
        
        # Get API key from environment
        self.api_key = os.getenv('LLM_API_KEY') or os.getenv('GEMINI_API_KEY')
        
        # Extract session_id from config
        self.session_id = self.config.get('session_id', 'unknown')
        
        # Validate video path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Generate report
        if GEMINI_AVAILABLE and self.api_key:
            report = self._generate_with_gemini(video_path, system_instruction, user_query)
        else:
            report = self._generate_mock_report(video_path)
        
        # Add metadata
        report['session_id'] = self.session_id
        report['generated_at'] = datetime.now(timezone.utc).astimezone().isoformat()
        
        # Store for saving
        self.metadata = report
        
        return (report,)
    
    def _generate_with_gemini(self, video_path: str, system_instruction: str, user_query: str) -> REPORT:
        """Generate report using Google Gemini API"""
        print(f"Analyzing video with Gemini: {video_path}")
        
        # Configure API
        genai.configure(api_key=self.api_key)
        
        # Upload video
        print("Uploading video to Gemini...")
        uploaded_file = genai.upload_file(path=video_path)
        print(f"Uploaded: {uploaded_file.display_name}")
        
        # Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            print("Waiting for file processing...")
            time.sleep(5)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            raise RuntimeError("File processing failed")
        
        # Generate response
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction
        )
        
        response = model.generate_content([
            {
                "role": "user",
                "parts": [
                    {"file_data": {"file_uri": uploaded_file.uri, "mime_type": uploaded_file.mime_type}},
                    {"text": user_query}
                ]
            }
        ])
        
        # Parse response
        parsed = self._parse_response(response.text)
        
        # Cleanup
        try:
            genai.delete_file(uploaded_file.name)
        except:
            pass
        
        return parsed
    
    def _parse_response(self, response_text: str) -> REPORT:
        """Parse LLM response into standardized report format"""
        report_marker = "### COMPREHENSIVE REPORT ###"
        summary_marker = "### SUMMARY FOR USER ###"
        
        report_start = response_text.find(report_marker)
        summary_start = response_text.find(summary_marker)
        
        if report_start == -1 or summary_start == -1:
            # Fallback parsing
            comprehensive_report = response_text
            summary = self._extract_summary_fallback(response_text)
        else:
            comprehensive_report = response_text[report_start + len(report_marker):summary_start].strip()
            summary_text = response_text[summary_start + len(summary_marker):].strip()
            summary = self._parse_summary(summary_text)
        
        return {
            "session_id": self.session_id,
            "comprehensive_report": comprehensive_report,
            "summary_for_user": summary,
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat()
        }
    
    def _parse_summary(self, summary_text: str) -> Dict:
        """Parse summary section into structured format"""
        lines = summary_text.split('\n')
        
        risk_level = "Medium"
        justification = ""
        actions = []
        
        in_actions = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("OVERALL RISK LEVEL:"):
                risk_level = line.split(":", 1)[1].strip()
            elif line.startswith("JUSTIFICATION:"):
                justification = line.split(":", 1)[1].strip()
            elif line.startswith("RECOMMENDED ACTIONS:"):
                in_actions = True
            elif in_actions and line and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                actions.append(line.lstrip("-•* ").strip())
            elif in_actions and line and not line.startswith("OVERALL") and not line.startswith("JUSTIFICATION"):
                # Multi-line justification or action
                if not actions:
                    justification += " " + line
                else:
                    actions[-1] += " " + line
        
        return {
            "overall_risk_level": risk_level,
            "justification": justification,
            "recommended_actions": actions
        }
    
    def _extract_summary_fallback(self, text: str) -> Dict:
        """Fallback summary extraction"""
        return {
            "overall_risk_level": "Medium",
            "justification": "Automated analysis completed. Manual review recommended.",
            "recommended_actions": ["Review full report", "Verify findings"]
        }
    
    def _generate_mock_report(self, video_path: str) -> REPORT:
        """Generate mock report for testing"""
        print(f"Generating mock report for: {video_path}")
        
        comprehensive = """### COMPREHENSIVE REPORT ###
TIMESTAMP: 00:00:00 - 00:00:05
OBSERVATIONS: Video surveillance footage shows main entrance area. Two individuals present in frame. Normal foot traffic observed.
FLAGGED_ITEMS: None
---
TIMESTAMP: 00:00:05 - 00:00:10
OBSERVATIONS: Person wearing dark clothing enters from left side. Another person exits through main door. Movement patterns appear normal.
FLAGGED_ITEMS: None
---"""
        
        summary = {
            "overall_risk_level": "Low",
            "justification": "Mock analysis indicates normal activity patterns. No immediate threats detected in the analyzed footage.",
            "recommended_actions": [
                "Continue routine monitoring",
                "Archive footage for record keeping",
                "Replace with actual VLM analysis by configuring GEMINI_API_KEY"
            ]
        }
        
        return {
            "session_id": self.session_id,
            "comprehensive_report": comprehensive,
            "summary_for_user": summary,
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat()
        }
    
    def _write_metadata(self, metadata_dir: Path) -> None:
        """Write report.json and vector data"""
        # Save report.json
        report_path = metadata_dir / "report.json"
        self._save_json(report_path, self.metadata)
        print(f"Saved report to {report_path}")
        
        # Save vector data (JSONL format for potential vector DB ingestion)
        vectors_dir = metadata_dir.parent / "vectors"
        vectors_dir.mkdir(exist_ok=True)
        
        vector_path = vectors_dir / "report_vectors.jsonl"
        
        # Create vector-ready entries
        with open(vector_path, 'w', encoding='utf-8') as f:
            # Entry for comprehensive report
            f.write(self._format_jsonl({
                "text": self.metadata['comprehensive_report'],
                "metadata": {
                    "session_id": self.session_id,
                    "type": "comprehensive_report",
                    "generated_at": self.metadata['generated_at']
                }
            }))
            f.write('\n')
            
            # Entry for summary
            f.write(self._format_jsonl({
                "text": f"Risk Level: {self.metadata['summary_for_user']['overall_risk_level']}. {self.metadata['summary_for_user']['justification']}",
                "metadata": {
                    "session_id": self.session_id,
                    "type": "summary",
                    "risk_level": self.metadata['summary_for_user']['overall_risk_level'],
                    "generated_at": self.metadata['generated_at']
                }
            }))
        
        print(f"Saved vector data to {vector_path}")
    
    def _format_jsonl(self, data: Dict) -> str:
        """Format data as JSONL entry"""
        import json
        return json.dumps(data, ensure_ascii=False)
