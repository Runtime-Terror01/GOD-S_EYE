"""
VLM Report Generation Node
Generates comprehensive security reports using Google Gemini AI
"""

import os
import time
import json
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

try:
    from google import genai
    from google.genai.errors import APIError
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: Google GenAI library not available. Install with: pip install google-generativeai")

class VLMReportNode:
    def __init__(self):
        self.model_name = "gemini-flash-latest"
        
        # Initialize client if available
        self.client = None
        if GENAI_AVAILABLE:
            try:
                # Get API key from environment variable for security
                api_key = os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    # Fallback to hardcoded key (not recommended for production)
                    api_key = "AIzaSyCLuGKM6LHq6rb8tcCo5CWgqxbG5yswqeY"
                    print("Warning: Using hardcoded API key. Set GOOGLE_API_KEY environment variable for better security.")
                
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"Failed to initialize Google GenAI client: {e}")
                self.client = None
        
        # AI Prompting
        self.system_instruction = (
            "You are a highly specialized Security Footage Analysis Agent. Your task is to analyze video footage "
            "and generate a two-part security report. Your response MUST be objective, factual, and strictly "
            "adhere to the requested format. Do not add any conversational text outside of the requested structure."
        )
        
        self.user_query = (
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
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alert_triggered": ("BOOLEAN",),
                "frame_metadata": ("FRAME_METADATA",),
                "threat_analysis": ("STRING",),
            },
            "optional": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Direct path to video file for VLM analysis"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("comprehensive_report", "executive_summary")
    FUNCTION = "generate_report"
    CATEGORY = "Threat Detection"
    
    def _parse_model_response(self, response_text: str) -> Optional[List[str]]:
        """Parse the model's response into report and summary sections"""
        try:
            report_marker = "### COMPREHENSIVE REPORT ###"
            summary_marker = "### SUMMARY FOR USER ###"
            report_start_index = response_text.find(report_marker)
            summary_start_index = response_text.find(summary_marker)

            if report_start_index == -1 or summary_start_index == -1:
                return None

            report_content = response_text[report_start_index + len(report_marker):summary_start_index].strip()
            summary_content = response_text[summary_start_index + len(summary_marker):].strip()

            if not report_content or not summary_content:
                return None

            return [report_content, summary_content]
        except Exception:
            return None
    
    def create_video_from_frames(self, frame_metadata: dict) -> str:
        """Create a video file from extracted frames for VLM analysis"""
        
        try:
            import cv2
            
            frames_info = frame_metadata['frames']
            if not frames_info:
                raise ValueError("No frames found in metadata")
            
            # Create output video path
            session_id = frame_metadata['session_id']
            output_dir = frame_metadata['storage_directory']
            output_video_path = os.path.join(output_dir, f"reconstructed_video_{session_id}.mp4")
            
            # Get video properties from first frame
            first_frame_path = frames_info[0]['path']
            if not os.path.exists(first_frame_path):
                raise ValueError(f"First frame not found: {first_frame_path}")
            
            # Read first frame to get dimensions
            first_frame = cv2.imread(first_frame_path)
            height, width, _ = first_frame.shape
            
            # Initialize video writer
            fps = frame_metadata.get('fps', 30.0)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            print(f"Creating video from {len(frames_info)} frames...")
            
            # Write frames to video
            for i, frame_info in enumerate(frames_info):
                frame_path = frame_info['path']
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    video_writer.write(frame)
                else:
                    print(f"Warning: Frame not found: {frame_path}")
            
            video_writer.release()
            
            if os.path.exists(output_video_path):
                print(f"✓ Video created: {output_video_path}")
                return output_video_path
            else:
                raise ValueError("Failed to create video file")
                
        except ImportError:
            raise ValueError("OpenCV not available for video creation")
        except Exception as e:
            raise ValueError(f"Failed to create video from frames: {e}")
    
    def analyze_video_with_vlm(self, video_path: str) -> List[str]:
        """Analyze video using Google Gemini VLM"""
        
        if not GENAI_AVAILABLE or not self.client:
            raise ValueError("Google GenAI not available or not properly initialized")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at path: {video_path}")

        uploaded_file = None
        try:
            print(f"Uploading video to Google AI: {video_path}")
            
            # Upload the file
            uploaded_file = self.client.files.upload(file=video_path)

            # Wait for processing to complete
            print("Waiting for video processing...")
            while True:
                f_status = self.client.files.get(name=uploaded_file.name)
                if f_status.state == "ACTIVE":
                    break
                elif f_status.state == "FAILED":
                    raise Exception(f"File processing failed for file: {uploaded_file.name}")
                time.sleep(5)

            print("Generating security report...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"file_data": {"file_uri": uploaded_file.uri, "mime_type": uploaded_file.mime_type}},
                            {"text": self.user_query},
                        ],
                    }
                ],
                config=genai.types.GenerateContentConfig(system_instruction=self.system_instruction)
            )

            parsed_data = self._parse_model_response(response.text)
            if not parsed_data:
                raise Exception("Failed to parse the model's response. The output may be malformed.")
                
            return parsed_data

        except APIError as e:
            raise e
        except Exception as e:
            raise Exception(f"An unexpected error occurred during video analysis: {e}")

        finally:
            # Cleanup uploaded file
            if uploaded_file:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    print("✓ Cleaned up uploaded file")
                except Exception as e:
                    print(f"Warning: Failed to cleanup uploaded file: {e}")
    
    def generate_fallback_report(self, frame_metadata: dict, threat_analysis: str) -> Tuple[str, str]:
        """Generate a fallback report when VLM is not available"""
        
        try:
            threat_data = json.loads(threat_analysis)
        except:
            threat_data = {}
        
        # Generate comprehensive report
        report_parts = []
        report_parts.append("=== AUTOMATED SECURITY ANALYSIS ===")
        report_parts.append(f"Analysis ID: {str(uuid.uuid4())}")
        report_parts.append(f"Timestamp: {datetime.now().isoformat()}")
        report_parts.append("")
        
        # Video information
        report_parts.append("VIDEO INFORMATION:")
        report_parts.append(f"Source: {frame_metadata.get('video_source', 'Unknown')}")
        report_parts.append(f"Duration: {frame_metadata.get('duration', 0):.1f} seconds")
        report_parts.append(f"Frames analyzed: {frame_metadata.get('total_frames_extracted', 0)}")
        report_parts.append(f"Resolution: {frame_metadata.get('resolution', {}).get('width', 0)}x{frame_metadata.get('resolution', {}).get('height', 0)}")
        report_parts.append("")
        
        # Threat analysis
        if threat_data.get('threats_detected'):
            report_parts.append("THREATS DETECTED:")
            for threat_class in threat_data.get('threats_detected', []):
                threat_info = threat_data.get('detailed_analysis', {}).get(threat_class, {})
                report_parts.append(f"• {threat_class.upper()}: {threat_info.get('total_detections', 0)} detections over {threat_info.get('total_duration', 0):.1f}s")
        else:
            report_parts.append("THREATS DETECTED: None")
        
        comprehensive_report = "\n".join(report_parts)
        
        # Generate executive summary
        threat_level = threat_data.get('threat_level', 'LOW')
        summary_parts = []
        summary_parts.append(f"OVERALL RISK LEVEL: {threat_level}")
        summary_parts.append("")
        
        if threat_data.get('threats_detected'):
            summary_parts.append("JUSTIFICATION: Automated threat detection identified potential security concerns in the video footage.")
            summary_parts.append("")
            summary_parts.append("RECOMMENDED ACTIONS:")
            summary_parts.append("• Review flagged video segments manually")
            summary_parts.append("• Verify threat detections with security personnel")
            summary_parts.append("• Consider additional security measures if threats are confirmed")
        else:
            summary_parts.append("JUSTIFICATION: No significant threats detected in automated analysis.")
            summary_parts.append("")
            summary_parts.append("RECOMMENDED ACTIONS:")
            summary_parts.append("• Archive for informational purposes")
            summary_parts.append("• No immediate action required")
        
        executive_summary = "\n".join(summary_parts)
        
        return comprehensive_report, executive_summary
    
    def generate_report(self, alert_triggered: bool, frame_metadata: dict, threat_analysis: str, video_path: str = "") -> Tuple[str, str]:
        """Main report generation function"""
        
        print(f"\n=== VLM Report Generation ===")
        print(f"Alert triggered: {alert_triggered}")
        print(f"VLM available: {GENAI_AVAILABLE and self.client is not None}")
        
        # Only generate VLM report if alert is triggered
        if not alert_triggered:
            print("No alert triggered - generating basic summary")
            return self.generate_fallback_report(frame_metadata, threat_analysis)
        
        # Try to use VLM if available
        if GENAI_AVAILABLE and self.client:
            try:
                # Determine video path for analysis
                analysis_video_path = video_path
                
                if not analysis_video_path or not os.path.exists(analysis_video_path):
                    # Try to get original video path from metadata
                    analysis_video_path = frame_metadata.get('video_source', '')
                
                if not analysis_video_path or not os.path.exists(analysis_video_path):
                    # Create video from frames
                    print("Creating video from extracted frames for VLM analysis...")
                    analysis_video_path = self.create_video_from_frames(frame_metadata)
                
                # Analyze with VLM
                print(f"Analyzing video with VLM: {analysis_video_path}")
                vlm_results = self.analyze_video_with_vlm(analysis_video_path)
                
                print("✓ VLM analysis complete")
                return vlm_results[0], vlm_results[1]
                
            except Exception as e:
                print(f"VLM analysis failed: {e}")
                print("Falling back to automated report generation")
                return self.generate_fallback_report(frame_metadata, threat_analysis)
        
        else:
            print("VLM not available - generating automated report")
            return self.generate_fallback_report(frame_metadata, threat_analysis)

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VLMReportNode": VLMReportNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMReportNode": "VLM Security Report"
}