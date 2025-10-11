"""
Alert Node
Intelligent threat detection and alert generation with temporal analysis
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import math

class AlertNode:
    def __init__(self):
        # Alert configuration
        self.threat_classes = {
            1: "gun",
            2: "explosive", 
            3: "grenade",
            4: "knife"
        }
        
        # Temporal analysis parameters
        self.temporal_window = 3.0  # seconds
        self.min_confidence_threshold = 0.6
        self.spatial_distance_threshold = 100  # pixels
        self.min_consecutive_detections = 3
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "predictions_json": ("STRING",),
                "alert_sensitivity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Higher values = more sensitive to threats"
                }),
                "temporal_grouping": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Time window (seconds) for grouping detections"
                }),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("alert_triggered", "alert_summary", "threat_analysis")
    FUNCTION = "analyze_threats"
    CATEGORY = "Threat Detection"
    
    def calculate_spatial_distance(self, detection1: Dict, detection2: Dict) -> float:
        """Calculate spatial distance between two detections"""
        x1, y1 = detection1['x'], detection1['y']
        x2, y2 = detection2['x'], detection2['y']
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def group_detections_by_proximity(self, detections: List[Dict], spatial_threshold: float) -> List[List[Dict]]:
        """Group detections that are spatially close to each other"""
        if not detections:
            return []
        
        groups = []
        used_detections = set()
        
        for i, detection in enumerate(detections):
            if i in used_detections:
                continue
                
            # Start a new group
            current_group = [detection]
            used_detections.add(i)
            
            # Find nearby detections
            for j, other_detection in enumerate(detections):
                if j in used_detections or i == j:
                    continue
                    
                distance = self.calculate_spatial_distance(detection, other_detection)
                if distance <= spatial_threshold:
                    current_group.append(other_detection)
                    used_detections.add(j)
            
            groups.append(current_group)
        
        return groups
    
    def analyze_temporal_patterns(self, predictions: List[Dict], temporal_window: float) -> Dict[str, Any]:
        """Analyze temporal patterns in threat detections"""
        
        # Group predictions by class and time
        class_timelines = {}
        
        for prediction in predictions:
            frame_ref = prediction.get('frame_reference', {})
            timestamp = frame_ref.get('timestamp', 0)
            
            for detection in prediction.get('predictions', []):
                class_name = detection['class']
                confidence = detection['confidence']
                
                if class_name not in class_timelines:
                    class_timelines[class_name] = []
                
                class_timelines[class_name].append({
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'detection': detection,
                    'frame_id': frame_ref.get('frame_id'),
                    'frame_index': frame_ref.get('frame_index', 0)
                })
        
        # Analyze each class timeline
        threat_analysis = {}
        
        for class_name, timeline in class_timelines.items():
            # Sort by timestamp
            timeline.sort(key=lambda x: x['timestamp'])
            
            # Find continuous threat periods
            threat_periods = []
            current_period = []
            
            for i, detection in enumerate(timeline):
                if not current_period:
                    current_period = [detection]
                else:
                    # Check if this detection is within temporal window of the last one
                    time_diff = detection['timestamp'] - current_period[-1]['timestamp']
                    
                    if time_diff <= temporal_window:
                        current_period.append(detection)
                    else:
                        # End current period and start new one
                        if len(current_period) >= self.min_consecutive_detections:
                            threat_periods.append(current_period)
                        current_period = [detection]
            
            # Don't forget the last period
            if len(current_period) >= self.min_consecutive_detections:
                threat_periods.append(current_period)
            
            # Analyze threat periods
            if threat_periods:
                max_confidence = max(d['confidence'] for period in threat_periods for d in period)
                total_duration = sum(
                    period[-1]['timestamp'] - period[0]['timestamp'] 
                    for period in threat_periods
                )
                avg_confidence = sum(d['confidence'] for period in threat_periods for d in period) / sum(len(period) for period in threat_periods)
                
                threat_analysis[class_name] = {
                    'threat_periods': len(threat_periods),
                    'max_confidence': max_confidence,
                    'avg_confidence': avg_confidence,
                    'total_duration': total_duration,
                    'total_detections': sum(len(period) for period in threat_periods),
                    'periods_data': threat_periods
                }
        
        return threat_analysis
    
    def calculate_threat_score(self, threat_analysis: Dict, alert_sensitivity: float) -> Tuple[float, Dict]:
        """Calculate overall threat score and detailed scoring"""
        
        if not threat_analysis:
            return 0.0, {}
        
        # Threat class severity weights
        severity_weights = {
            'gun': 1.0,
            'explosive': 1.0,
            'grenade': 0.9,
            'knife': 0.7
        }
        
        total_score = 0.0
        detailed_scores = {}
        
        for class_name, analysis in threat_analysis.items():
            # Base score from confidence
            confidence_score = analysis['avg_confidence'] * 0.4
            
            # Duration score (longer presence = higher threat)
            duration_score = min(analysis['total_duration'] / 10.0, 1.0) * 0.3
            
            # Consistency score (more detections = higher threat)
            consistency_score = min(analysis['total_detections'] / 20.0, 1.0) * 0.3
            
            # Combine scores
            class_score = (confidence_score + duration_score + consistency_score) * severity_weights.get(class_name, 0.5)
            
            # Apply sensitivity multiplier
            class_score *= alert_sensitivity
            
            detailed_scores[class_name] = {
                'confidence_score': confidence_score,
                'duration_score': duration_score,
                'consistency_score': consistency_score,
                'final_score': class_score,
                'severity_weight': severity_weights.get(class_name, 0.5)
            }
            
            total_score += class_score
        
        # Normalize total score
        total_score = min(total_score, 1.0)
        
        return total_score, detailed_scores
    
    def generate_alert_summary(self, threat_analysis: Dict, threat_score: float, detailed_scores: Dict) -> str:
        """Generate human-readable alert summary"""
        
        if threat_score < 0.3:
            return "No significant threats detected."
        
        summary_parts = []
        
        # Overall threat level
        if threat_score >= 0.8:
            threat_level = "CRITICAL"
        elif threat_score >= 0.6:
            threat_level = "HIGH"
        elif threat_score >= 0.4:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"
        
        summary_parts.append(f"THREAT LEVEL: {threat_level} (Score: {threat_score:.2f})")
        
        # Detailed threat breakdown
        for class_name, analysis in threat_analysis.items():
            periods = analysis['threat_periods']
            duration = analysis['total_duration']
            max_conf = analysis['max_confidence']
            detections = analysis['total_detections']
            
            summary_parts.append(
                f"• {class_name.upper()}: {periods} threat period(s), "
                f"{duration:.1f}s total duration, "
                f"{detections} detections, "
                f"max confidence: {max_conf:.2f}"
            )
        
        return "\n".join(summary_parts)
    
    def generate_threat_analysis_json(self, threat_analysis: Dict, detailed_scores: Dict, threat_score: float) -> str:
        """Generate detailed threat analysis in JSON format"""
        
        analysis_data = {
            'analysis_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'overall_threat_score': threat_score,
            'threat_level': 'CRITICAL' if threat_score >= 0.8 else 
                          'HIGH' if threat_score >= 0.6 else 
                          'MEDIUM' if threat_score >= 0.4 else 'LOW',
            'threats_detected': list(threat_analysis.keys()),
            'detailed_analysis': {}
        }
        
        for class_name, analysis in threat_analysis.items():
            analysis_data['detailed_analysis'][class_name] = {
                'threat_periods': analysis['threat_periods'],
                'total_detections': analysis['total_detections'],
                'max_confidence': analysis['max_confidence'],
                'avg_confidence': analysis['avg_confidence'],
                'total_duration': analysis['total_duration'],
                'scoring': detailed_scores.get(class_name, {}),
                'first_detection_time': analysis['periods_data'][0][0]['timestamp'] if analysis['periods_data'] else 0,
                'last_detection_time': analysis['periods_data'][-1][-1]['timestamp'] if analysis['periods_data'] else 0
            }
        
        return json.dumps(analysis_data, indent=2)
    
    def analyze_threats(self, predictions_json: str, alert_sensitivity: float, temporal_grouping: float) -> Tuple[bool, str, str]:
        """Main threat analysis function"""
        
        print(f"\n=== Threat Alert Analysis ===")
        print(f"Alert sensitivity: {alert_sensitivity}")
        print(f"Temporal grouping: {temporal_grouping}s")
        
        try:
            # Parse predictions JSON
            predictions = json.loads(predictions_json)
            
            if not predictions:
                return (False, "No predictions to analyze.", "{}")
            
            print(f"Analyzing {len(predictions)} prediction frames...")
            
            # Update temporal window based on user input
            self.temporal_window = temporal_grouping
            
            # Analyze temporal patterns
            threat_analysis = self.analyze_temporal_patterns(predictions, temporal_grouping)
            
            if not threat_analysis:
                return (False, "No significant threat patterns detected.", "{}")
            
            # Calculate threat score
            threat_score, detailed_scores = self.calculate_threat_score(threat_analysis, alert_sensitivity)
            
            # Determine if alert should be triggered
            alert_triggered = threat_score >= 0.4  # Minimum threshold for alert
            
            # Generate summaries
            alert_summary = self.generate_alert_summary(threat_analysis, threat_score, detailed_scores)
            threat_analysis_json = self.generate_threat_analysis_json(threat_analysis, detailed_scores, threat_score)
            
            print(f"✓ Analysis complete: Alert triggered = {alert_triggered}, Threat score = {threat_score:.2f}")
            
            return (alert_triggered, alert_summary, threat_analysis_json)
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse predictions JSON: {e}"
            print(f"Error: {error_msg}")
            return (False, error_msg, "{}")
        
        except Exception as e:
            error_msg = f"Error during threat analysis: {e}"
            print(f"Error: {error_msg}")
            return (False, error_msg, "{}")

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AlertNode": AlertNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlertNode": "Threat Alert Analysis"
}