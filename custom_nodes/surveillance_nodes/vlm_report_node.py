"""
VLM Report Node for Comprehensive Video Analysis
Integrates with Vision-Language Models for scene understanding and report generation
"""

import os
import json
import uuid
import base64
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import asyncio
import hashlib
from collections import defaultdict

# API Client imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Generative AI not available. Install with: pip install google-generativeai")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

# For embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: Sentence transformers not available. Install with: pip install sentence-transformers")

class VLMReportNode:
    """Generate comprehensive reports using Vision-Language Models"""
    
    # Risk level thresholds based on detected elements
    RISK_SCORING = {
        'weapon_detected': 50,
        'suspicious_behavior': 30,
        'abandoned_object': 40,
        'crowd_gathering': 20,
        'restricted_area_breach': 45,
        'face_recognition_match': 35,
        'vehicle_suspicious': 25,
        'rapid_movement': 15,
        'loitering': 20,
        'concealment_behavior': 30
    }
    
    def __init__(self):
        self.vlm_provider = None
        self.vlm_client = None
        self.embedding_model = None
        self.api_key = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_metadata": ("FRAME_METADATA",),
                "detection_metadata": ("DETECTION_METADATA",),
                "track_metadata": ("TRACK_METADATA",),
                "vlm_provider": (["gemini-2.0-flash", "gpt-4-vision", "claude-3-opus"], {
                    "default": "gemini-2.0-flash",
                    "tooltip": "Vision-Language Model provider"
                }),
                "analysis_mode": (["comprehensive", "summary", "threat_focused", "behavioral"], {
                    "default": "comprehensive",
                    "tooltip": "Type of analysis to perform"
                }),
                "max_frames_to_analyze": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 100,
                    "tooltip": "Maximum frames to send to VLM"
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key for VLM provider (uses env var if empty)"
                }),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom analysis prompt"
                }),
                "alert_data": ("ALERT_DATA",),
                "generate_embeddings": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate embeddings for semantic search"
                }),
            }
        }
    
    RETURN_TYPES = ("VLM_REPORT", "STRING", "STRING", "EMBEDDINGS")
    RETURN_NAMES = ("report", "comprehensive_analysis", "executive_summary", "embeddings")
    FUNCTION = "generate_report"
    CATEGORY = "Surveillance/Analysis"
    
    def _initialize_vlm(self, provider: str, api_key: str = ""):
        """Initialize VLM client based on provider"""
        
        if provider.startswith("gemini"):
            if not GEMINI_AVAILABLE:
                raise RuntimeError("Google Generative AI not installed")
            
            # Get API key
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key required")
            
            genai.configure(api_key=api_key)
            self.vlm_client = genai.GenerativeModel(provider)
            self.vlm_provider = "gemini"
            
        elif provider.startswith("gpt-4"):
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI not installed")
            
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            self.vlm_client = OpenAI(api_key=api_key)
            self.vlm_provider = "openai"
            
        else:
            raise ValueError(f"Unsupported VLM provider: {provider}")
        
        print(f"✓ Initialized {provider} VLM")
    
    def _initialize_embedding_model(self):
        """Initialize embedding model for semantic search"""
        if EMBEDDINGS_AVAILABLE:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Initialized embedding model")
        else:
            print("⚠ Embedding model not available")
    
    def generate_report(self, frame_metadata: Dict, detection_metadata: Dict,
                       track_metadata: Dict, vlm_provider: str, analysis_mode: str,
                       max_frames_to_analyze: int, api_key: str = "",
                       custom_prompt: str = "", alert_data: Dict = None,
                       generate_embeddings: bool = True) -> Tuple[Dict, str, str, List]:
        """
        Generate comprehensive report using VLM
        
        Returns:
            report: Complete VLM report object
            comprehensive_analysis: Detailed timestamped analysis
            executive_summary: Brief summary
            embeddings: Embeddings for semantic search
        """
        
        # Initialize VLM if needed
        if self.vlm_client is None:
            self._initialize_vlm(vlm_provider, api_key)
        
        # Initialize embedding model if needed
        if generate_embeddings and self.embedding_model is None:
            self._initialize_embedding_model()
        
        # Prepare frames for analysis
        frames_to_analyze = self._select_key_frames(
            frame_metadata, detection_metadata, track_metadata, 
            max_frames_to_analyze, alert_data
        )
        
        # Generate analysis prompt
        prompt = self._generate_analysis_prompt(
            analysis_mode, detection_metadata, track_metadata, 
            alert_data, custom_prompt
        )
        
        # Analyze video with VLM
        print(f"Analyzing {len(frames_to_analyze)} key frames with {vlm_provider}...")
        
        vlm_response = self._call_vlm_api(frames_to_analyze, prompt)
        
        # Parse VLM response
        parsed_response = self._parse_vlm_response(vlm_response)
        
        # Generate comprehensive analysis
        comprehensive_analysis = self._generate_comprehensive_analysis(
            parsed_response, detection_metadata, track_metadata, alert_data
        )
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            parsed_response, detection_metadata, track_metadata, alert_data
        )
        
        # Calculate risk assessment
        risk_assessment = self._calculate_risk_assessment(
            parsed_response, detection_metadata, track_metadata, alert_data
        )
        
        # Extract key events
        key_events = self._extract_key_events(
            parsed_response, detection_metadata, track_metadata
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_assessment, parsed_response, alert_data
        )
        
        # Create report object
        report = {
            'report_id': str(uuid.uuid4()),
            'session_id': frame_metadata['session_id'],
            'timestamp': datetime.now().isoformat(),
            'video_source': frame_metadata['video_source'],
            'analysis_mode': analysis_mode,
            'vlm_provider': vlm_provider,
            'frames_analyzed': len(frames_to_analyze),
            'total_frames': frame_metadata['total_frames_extracted'],
            'comprehensive_report': comprehensive_analysis,
            'executive_summary': executive_summary,
            'risk_assessment': risk_assessment,
            'key_events': key_events,
            'recommendations': recommendations,
            'detection_summary': self._summarize_detections(detection_metadata),
            'tracking_summary': self._summarize_tracks(track_metadata),
            'alert_summary': self._summarize_alerts(alert_data) if alert_data else None,
            'raw_vlm_response': vlm_response,
            'metadata': {
                'duration': frame_metadata.get('duration', 0),
                'fps': frame_metadata.get('fps', 0),
                'resolution': frame_metadata.get('video_properties', {}).get('resolution', {})
            }
        }
        
        # Generate embeddings for semantic search
        embeddings = []
        if generate_embeddings and self.embedding_model:
            embeddings = self._generate_embeddings(report)
        
        # Save report
        self._save_report(report, frame_metadata['storage_directory'])
        
        print(f"✓ Generated VLM report with risk level: {risk_assessment['level']}")
        
        return (report, comprehensive_analysis, executive_summary, embeddings)
    
    def _select_key_frames(self, frame_metadata: Dict, detection_metadata: Dict,
                          track_metadata: Dict, max_frames: int, 
                          alert_data: Dict = None) -> List[Dict]:
        """Select most important frames for VLM analysis"""
        
        key_frames = []
        frame_scores = {}
        
        # Get all available frames
        all_frames = frame_metadata.get('frames', [])
        
        # Score frames based on detections
        threat_frames = set(detection_metadata.get('threat_analysis', {}).get('threat_frames', []))
        detection_map = detection_metadata.get('frame_detections_map', {})
        
        for frame in all_frames:
            frame_id = frame['frame_id']
            score = 0
            
            # High score for threat frames
            if frame_id in threat_frames:
                score += 100
            
            # Score based on number of detections
            if frame_id in detection_map:
                detections = detection_map[frame_id]
                score += len(detections) * 10
                
                # Extra score for specific classes
                for det in detections:
                    if det['class_name'] in ['person', 'car', 'knife', 'gun']:
                        score += 20
            
            # Score based on tracks
            for track in track_metadata.get('all_tracks', []):
                if frame['frame_index'] in track['attributes'].get('frame_indices', []):
                    score += 5
            
            # Score based on alerts
            if alert_data:
                if frame_id in alert_data.get('evidence', {}).get('frame_ids', []):
                    score += 150
            
            frame_scores[frame_id] = score
        
        # Sort frames by score and select top N
        sorted_frames = sorted(all_frames, 
                              key=lambda f: frame_scores.get(f['frame_id'], 0), 
                              reverse=True)
        
        # Select frames with good temporal distribution
        selected_indices = set()
        for frame in sorted_frames:
            if len(key_frames) >= max_frames:
                break
            
            # Ensure temporal distribution
            frame_idx = frame['frame_index']
            too_close = any(abs(frame_idx - idx) < 5 for idx in selected_indices)
            
            if not too_close or frame_scores[frame['frame_id']] > 100:
                key_frames.append(frame)
                selected_indices.add(frame_idx)
        
        # Sort by timestamp for chronological analysis
        key_frames.sort(key=lambda f: f['timestamp'])
        
        return key_frames
    
    def _generate_analysis_prompt(self, mode: str, detection_metadata: Dict,
                                 track_metadata: Dict, alert_data: Dict,
                                 custom_prompt: str) -> str:
        """Generate appropriate prompt based on analysis mode"""
        
        base_prompt = """You are an expert security analyst reviewing surveillance footage. 
        Analyze the provided frames and generate a detailed security assessment.
        
        Context:
        - Total detections: {total_detections}
        - Classes detected: {classes}
        - Number of tracks: {num_tracks}
        - Suspicious patterns detected: {suspicious_patterns}
        
        """
        
        # Fill in context
        context_data = {
            'total_detections': detection_metadata.get('total_detections', 0),
            'classes': ', '.join(detection_metadata.get('class_distribution', {}).keys()),
            'num_tracks': track_metadata.get('total_tracks', 0),
            'suspicious_patterns': ', '.join(track_metadata.get('suspicious_analysis', {}).keys())
        }
        
        prompt = base_prompt.format(**context_data)
        
        # Mode-specific instructions
        if mode == "comprehensive":
            prompt += """
            Provide a COMPREHENSIVE analysis including:
            1. **Timeline Analysis**: What happens at each timestamp (be specific about times)
            2. **Object Detection**: All objects and people observed
            3. **Behavioral Analysis**: Movement patterns, interactions, suspicious behaviors
            4. **Threat Assessment**: Any security concerns or risks identified
            5. **Environmental Context**: Location details, time of day, visibility conditions
            
            Format your response with clear timestamps like:
            [00:00-00:05] Description of events...
            [00:05-00:10] Description of events...
            """
            
        elif mode == "threat_focused":
            prompt += """
            Focus ONLY on potential security threats:
            1. Identify any weapons, suspicious objects, or dangerous items
            2. Flag abnormal behaviors (loitering, concealment, erratic movement)
            3. Note unauthorized access or restricted area breaches
            4. Assess crowd dynamics and density risks
            5. Rate overall threat level: NONE, LOW, MEDIUM, HIGH, CRITICAL
            """
            
        elif mode == "behavioral":
            prompt += """
            Analyze BEHAVIORAL PATTERNS:
            1. Track individual movements and trajectories
            2. Identify group dynamics and interactions
            3. Detect anomalous behavior patterns
            4. Note any coordination between individuals
            5. Assess intent based on observed behavior
            """
            
        elif mode == "summary":
            prompt += """
            Provide a BRIEF SUMMARY (max 200 words):
            1. Main events observed
            2. Key security concerns
            3. Overall risk assessment
            4. Recommended actions
            """
        
        # Add alert context if available
        if alert_data and alert_data.get('threats_detected'):
            prompt += f"\n\nALERT CONTEXT: {', '.join(alert_data['threats_detected'])}"
            prompt += f"\nAlert Level: {alert_data.get('threat_level', 'UNKNOWN')}"
        
        # Add custom prompt if provided
        if custom_prompt:
            prompt += f"\n\nAdditional Instructions: {custom_prompt}"
        
        return prompt
    
    def _call_vlm_api(self, frames: List[Dict], prompt: str) -> str:
        """Call VLM API with frames and prompt"""
        
        if self.vlm_provider == "gemini":
            # Prepare frames for Gemini
            images = []
            for frame_data in frames:
                # Load frame image
                frame_path = frame_data['path']
                with open(frame_path, 'rb') as f:
                    image_data = f.read()
                
                # Create image part for Gemini
                image_part = {
                    'mime_type': 'image/jpeg',
                    'data': base64.b64encode(image_data).decode('utf-8')
                }
                images.append(image_part)
            
            # Create message with images and prompt
            message_parts = []
            for i, image in enumerate(images):
                timestamp = frames[i]['timestamp']
                message_parts.append(f"Frame at {timestamp:.2f}s:")
                message_parts.append(image)
            message_parts.append(prompt)
            
            # Generate response
            response = self.vlm_client.generate_content(message_parts)
            return response.text
            
        elif self.vlm_provider == "openai":
            # Prepare for GPT-4 Vision
            messages = [
                {"role": "system", "content": "You are an expert security analyst."},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            
            # Add images
            for frame_data in frames:
                with open(frame_data['path'], 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high"
                    }
                })
            
            response = self.vlm_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=4096
            )
            return response.choices[0].message.content
        
        return "VLM analysis not available"
    
    def _parse_vlm_response(self, response: str) -> Dict:
        """Parse VLM response into structured format"""
        
        parsed = {
            'timestamps': [],
            'events': [],
            'threats': [],
            'behaviors': [],
            'objects': [],
            'risk_indicators': []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse timestamp entries [00:00-00:05]
            if line.startswith('[') and ']' in line:
                timestamp_part = line[1:line.index(']')]
                description = line[line.index(']')+1:].strip()
                
                parsed['timestamps'].append({
                    'time_range': timestamp_part,
                    'description': description
                })
            
            # Detect section headers
            elif any(keyword in line.lower() for keyword in ['threat', 'risk', 'danger']):
                parsed['threats'].append(line)
            elif any(keyword in line.lower() for keyword in ['behavior', 'movement', 'activity']):
                parsed['behaviors'].append(line)
            elif any(keyword in line.lower() for keyword in ['object', 'person', 'vehicle']):
                parsed['objects'].append(line)
        
        return parsed
    
    def _generate_comprehensive_analysis(self, parsed_response: Dict,
                                        detection_metadata: Dict,
                                        track_metadata: Dict,
                                        alert_data: Dict) -> str:
        """Generate comprehensive timestamped analysis"""
        
        analysis = "# COMPREHENSIVE SURVEILLANCE ANALYSIS REPORT\n\n"
        analysis += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        analysis += "=" * 80 + "\n\n"
        
        # Executive Overview
        analysis += "## EXECUTIVE OVERVIEW\n"
        analysis += f"- Total Objects Detected: {detection_metadata.get('total_detections', 0)}\n"
        analysis += f"- Unique Tracks: {track_metadata.get('total_tracks', 0)}\n"
        analysis += f"- Threat Frames: {len(detection_metadata.get('threat_analysis', {}).get('threat_frames', []))}\n"
        
        if alert_data:
            analysis += f"- Alert Level: {alert_data.get('threat_level', 'NONE')}\n"
            analysis += f"- Threats Detected: {', '.join(alert_data.get('threats_detected', []))}\n"
        
        analysis += "\n" + "=" * 80 + "\n\n"
        
        # Temporal Analysis
        analysis += "## TEMPORAL ANALYSIS\n\n"
        
        for timestamp_entry in parsed_response.get('timestamps', []):
            analysis += f"**{timestamp_entry['time_range']}**\n"
            analysis += f"{timestamp_entry['description']}\n\n"
        
        # Object Analysis
        analysis += "\n## DETECTED OBJECTS & ENTITIES\n\n"
        
        class_dist = detection_metadata.get('class_distribution', {})
        for class_name, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
            threat_cat = "🔴 THREAT" if class_name in ['knife', 'gun', 'rifle'] else ""
            analysis += f"- {class_name}: {count} instances {threat_cat}\n"
        
        # Behavioral Analysis
        analysis += "\n## BEHAVIORAL PATTERNS\n\n"
        
        suspicious = track_metadata.get('suspicious_analysis', {})
        if suspicious.get('loitering'):
            analysis += "### Loitering Detected\n"
            for item in suspicious['loitering']:
                analysis += f"- Track {item['track_id']}: Duration {item['duration']:.1f}s, Severity: {item['severity']}\n"
        
        if suspicious.get('erratic_movement'):
            analysis += "\n### Erratic Movement\n"
            for item in suspicious['erratic_movement']:
                analysis += f"- Track {item['track_id']}: Variance {item['variance']:.2f}, Severity: {item['severity']}\n"
        
        if suspicious.get('abandoned_object'):
            analysis += "\n### Abandoned Objects\n"
            for item in suspicious['abandoned_object']:
                analysis += f"- {item['object_type']} (Track {item['track_id']}): Left for {item['duration']:.1f}s\n"
        
        # Threat Assessment
        analysis += "\n## THREAT ASSESSMENT\n\n"
        
        if parsed_response.get('threats'):
            for threat in parsed_response['threats']:
                analysis += f"- {threat}\n"
        else:
            analysis += "No immediate threats detected.\n"
        
        # VLM Insights
        analysis += "\n## AI SCENE UNDERSTANDING\n\n"
        if parsed_response.get('behaviors'):
            for behavior in parsed_response['behaviors'][:5]:
                analysis += f"- {behavior}\n"
        
        analysis += "\n" + "=" * 80 + "\n"
        
        return analysis
    
    def _generate_executive_summary(self, parsed_response: Dict,
                                   detection_metadata: Dict,
                                   track_metadata: Dict,
                                   alert_data: Dict) -> str:
        """Generate brief executive summary"""
        
        summary = "## EXECUTIVE SUMMARY\n\n"
        
        # Key statistics
        total_detections = detection_metadata.get('total_detections', 0)
        threat_frames = len(detection_metadata.get('threat_analysis', {}).get('threat_frames', []))
        threat_percentage = (threat_frames / detection_metadata.get('total_frames_processed', 1)) * 100
        
        summary += f"**Analysis Period**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"**Total Detections**: {total_detections} objects across {detection_metadata.get('frames_with_detections', 0)} frames\n"
        summary += f"**Threat Indicator**: {threat_percentage:.1f}% of frames contain potential threats\n\n"
        
        # Main findings
        summary += "**Key Findings**:\n"
        
        # Check for weapons
        threats = detection_metadata.get('threat_analysis', {}).get('detected_threats', [])
        if any(t in ['knife', 'gun', 'rifle'] for t in threats):
            summary += "⚠️ WEAPON DETECTED - Immediate action required\n"
        
        # Check suspicious behavior
        suspicious = track_metadata.get('suspicious_analysis', {})
        if suspicious.get('loitering'):
            summary += f"⚠️ Loitering behavior detected ({len(suspicious['loitering'])} instances)\n"
        
        if suspicious.get('abandoned_object'):
            summary += f"⚠️ Abandoned object(s) detected - potential security risk\n"
        
        # Overall assessment
        if alert_data:
            risk_level = alert_data.get('threat_level', 'UNKNOWN')
            summary += f"\n**Overall Risk Level**: {risk_level}\n"
            
            if risk_level in ['HIGH', 'CRITICAL']:
                summary += "\n**IMMEDIATE ACTION REQUIRED**\n"
                summary += "Security personnel should be dispatched immediately.\n"
        else:
            summary += "\n**Overall Risk Level**: LOW\n"
            summary += "No immediate security concerns detected.\n"
        
        # Recommendations
        summary += "\n**Recommendations**:\n"
        if threat_percentage > 50:
            summary += "1. Immediate security response required\n"
            summary += "2. Review footage for threat confirmation\n"
            summary += "3. Consider area lockdown\n"
        elif threat_percentage > 20:
            summary += "1. Increase monitoring of area\n"
            summary += "2. Deploy security patrol\n"
            summary += "3. Review recent activity logs\n"
        else:
            summary += "1. Continue routine monitoring\n"
            summary += "2. Log incident for records\n"
            summary += "3. No immediate action required\n"
        
        return summary
    
    def _calculate_risk_assessment(self, parsed_response: Dict,
                                  detection_metadata: Dict,
                                  track_metadata: Dict,
                                  alert_data: Dict) -> Dict:
        """Calculate overall risk assessment"""
        
        risk_score = 0
        risk_factors = []
        
        # Check for weapons
        threats = detection_metadata.get('threat_analysis', {}).get('detected_threats', [])
        if any(t in ['knife', 'gun', 'rifle'] for t in threats):
            risk_score += self.RISK_SCORING['weapon_detected']
            risk_factors.append('weapon_detected')
        
        # Check suspicious behavior
        suspicious = track_metadata.get('suspicious_analysis', {})
        if suspicious.get('loitering'):
            risk_score += self.RISK_SCORING['loitering']
            risk_factors.append('loitering')
        
        if suspicious.get('abandoned_object'):
            risk_score += self.RISK_SCORING['abandoned_object']
            risk_factors.append('abandoned_object')
        
        if suspicious.get('erratic_movement'):
            risk_score += self.RISK_SCORING['suspicious_behavior']
            risk_factors.append('erratic_movement')
        
        # Check crowd density
        crowd = detection_metadata.get('threat_analysis', {}).get('crowd_analysis', {})
        if crowd.get('risk_level') in ['high', 'critical']:
            risk_score += self.RISK_SCORING['crowd_gathering']
            risk_factors.append('crowd_risk')
        
        # Additional factors from alert
        if alert_data:
            if 'zone_intrusion' in alert_data.get('threats_detected', []):
                risk_score += self.RISK_SCORING['restricted_area_breach']
                risk_factors.append('zone_intrusion')
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = 'CRITICAL'
        elif risk_score >= 60:
            risk_level = 'HIGH'
        elif risk_score >= 40:
            risk_level = 'MEDIUM'
        elif risk_score >= 20:
            risk_level = 'LOW'
        else:
            risk_level = 'NONE'
        
        return {
            'level': risk_level,
            'score': risk_score,
            'factors': risk_factors,
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_key_events(self, parsed_response: Dict,
                           detection_metadata: Dict,
                           track_metadata: Dict) -> List[Dict]:
        """Extract key events from analysis"""
        
        key_events = []
        
        # Extract from timestamped entries
        for entry in parsed_response.get('timestamps', []):
            # Check if entry contains important keywords
            important_keywords = ['weapon', 'threat', 'suspicious', 'alert', 
                                 'abandoned', 'intrusion', 'unauthorized']
            
            if any(keyword in entry['description'].lower() for keyword in important_keywords):
                key_events.append({
                    'timestamp': entry['time_range'],
                    'event': entry['description'],
                    'priority': 'high'
                })
        
        # Add detection-based events
        threat_frames = detection_metadata.get('threat_analysis', {}).get('threat_frames', [])
        if threat_frames:
            key_events.append({
                'timestamp': 'multiple',
                'event': f"Potential threats detected in {len(threat_frames)} frames",
                'priority': 'high'
            })
        
        # Add tracking-based events
        suspicious = track_metadata.get('suspicious_analysis', {})
        for behavior_type, instances in suspicious.items():
            if instances:
                key_events.append({
                    'timestamp': 'various',
                    'event': f"{behavior_type.replace('_', ' ').title()}: {len(instances)} instances",
                    'priority': 'medium'
                })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        key_events.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
        
        return key_events[:10]  # Return top 10 events
    
    def _generate_recommendations(self, risk_assessment: Dict,
                                 parsed_response: Dict,
                                 alert_data: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        risk_level = risk_assessment['level']
        risk_factors = risk_assessment['factors']
        
        # Critical/High risk recommendations
        if risk_level in ['CRITICAL', 'HIGH']:
            recommendations.append("🚨 IMMEDIATE: Dispatch security personnel to location")
            recommendations.append("🚨 IMMEDIATE: Review live feeds and establish visual confirmation")
            
            if 'weapon_detected' in risk_factors:
                recommendations.append("🔴 Contact law enforcement immediately")
                recommendations.append("🔴 Initiate lockdown procedures if applicable")
                recommendations.append("🔴 Track and identify armed individual(s)")
            
            if 'abandoned_object' in risk_factors:
                recommendations.append("⚠️ Cordon off area around suspicious object")
                recommendations.append("⚠️ Consider bomb squad assessment")
                recommendations.append("⚠️ Evacuate immediate vicinity")
        
        # Medium risk recommendations
        elif risk_level == 'MEDIUM':
            recommendations.append("📋 Send security patrol to investigate")
            recommendations.append("📋 Increase monitoring frequency")
            recommendations.append("📋 Review access logs for anomalies")
            
            if 'loitering' in risk_factors:
                recommendations.append("👁️ Question individuals exhibiting loitering behavior")
                recommendations.append("👁️ Check for associated vehicles")
            
            if 'crowd_risk' in risk_factors:
                recommendations.append("👥 Deploy crowd control measures")
                recommendations.append("👥 Prepare medical teams")
        
        # Low risk recommendations
        elif risk_level == 'LOW':
            recommendations.append("✓ Continue standard monitoring procedures")
            recommendations.append("✓ Log incident for future reference")
            recommendations.append("✓ Review footage during next shift change")
        
        # No risk
        else:
            recommendations.append("✓ No action required - continue routine surveillance")
            recommendations.append("✓ Archive footage as per standard protocol")
        
        # Add specific recommendations based on VLM insights
        if parsed_response.get('threats'):
            recommendations.append("📊 Review AI-identified threats for validation")
        
        return recommendations
    
    def _summarize_detections(self, detection_metadata: Dict) -> Dict:
        """Summarize detection information"""
        return {
            'total_detections': detection_metadata.get('total_detections', 0),
            'frames_with_detections': detection_metadata.get('frames_with_detections', 0),
            'class_distribution': detection_metadata.get('class_distribution', {}),
            'threat_percentage': detection_metadata.get('threat_analysis', {}).get('threat_percentage', 0)
        }
    
    def _summarize_tracks(self, track_metadata: Dict) -> Dict:
        """Summarize tracking information"""
        return {
            'total_tracks': track_metadata.get('total_tracks', 0),
            'avg_track_length': track_metadata.get('track_statistics', {}).get('avg_track_length', 0),
            'suspicious_patterns': list(track_metadata.get('suspicious_analysis', {}).keys())
        }
    
    def _summarize_alerts(self, alert_data: Dict) -> Dict:
        """Summarize alert information"""
        if not alert_data:
            return None
        
        return {
            'alert_id': alert_data.get('alert_id'),
            'threat_level': alert_data.get('threat_level'),
            'threats_detected': alert_data.get('threats_detected', []),
            'requires_action': alert_data.get('requires_action', False)
        }
    
    def _generate_embeddings(self, report: Dict) -> List[Dict]:
        """Generate embeddings for semantic search"""
        
        if not self.embedding_model:
            return []
        
        embeddings_data = []
        
        # Split report into chunks
        chunks = []
        
        # Add executive summary as a chunk
        chunks.append({
            'text': report['executive_summary'],
            'type': 'executive_summary',
            'timestamp': report['timestamp']
        })
        
        # Split comprehensive report into sections
        sections = report['comprehensive_report'].split('\n##')
        for section in sections:
            if section.strip():
                chunks.append({
                    'text': section[:1000],  # Limit chunk size
                    'type': 'report_section',
                    'timestamp': report['timestamp']
                })
        
        # Add key events as chunks
        for event in report.get('key_events', []):
            chunks.append({
                'text': f"{event['timestamp']}: {event['event']}",
                'type': 'key_event',
                'timestamp': event.get('timestamp', report['timestamp'])
            })
        
        # Generate embeddings for each chunk
        for chunk in chunks:
            embedding = self.embedding_model.encode(chunk['text']).tolist()
            
            embeddings_data.append({
                'id': f"{report['report_id']}_{hashlib.md5(chunk['text'].encode()).hexdigest()[:8]}",
                'embedding': embedding,
                'text': chunk['text'],
                'metadata': {
                    'report_id': report['report_id'],
                    'session_id': report['session_id'],
                    'type': chunk['type'],
                    'timestamp': chunk['timestamp'],
                    'risk_level': report['risk_assessment']['level']
                }
            })
        
        print(f"✓ Generated {len(embeddings_data)} embeddings for semantic search")
        
        return embeddings_data
    
    def _save_report(self, report: Dict, storage_directory: str):
        """Save report to storage"""
        
        storage_dir = Path(storage_directory)
        reports_dir = storage_dir / "vlm_reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Save full report
        report_path = reports_dir / f"report_{report['report_id']}.json"
        with open(report_path, 'w') as f:
            # Create a copy without large fields for JSON storage
            save_report = {k: v for k, v in report.items() 
                          if k not in ['raw_vlm_response']}
            json.dump(save_report, f, indent=2)
        
        # Save comprehensive analysis as markdown
        analysis_path = reports_dir / f"analysis_{report['report_id']}.md"
        with open(analysis_path, 'w') as f:
            f.write(report['comprehensive_report'])
        
        # Save executive summary
        summary_path = reports_dir / f"summary_{report['report_id']}.md"
        with open(summary_path, 'w') as f:
            f.write(report['executive_summary'])
        
        print(f"✓ Report saved to {reports_dir}")

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VLMReportNode": VLMReportNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMReportNode": "VLM Scene Analysis & Reporting"
}