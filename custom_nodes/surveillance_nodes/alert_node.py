"""
Alert System Node with Comprehensive Threat Detection Logic
Analyzes detections, tracks, and patterns to generate security alerts
"""

import os
import json
import uuid
import requests
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import numpy as np

class AlertNode:
    """Generate security alerts based on detections and tracking data"""
    
    # Threat level definitions
    THREAT_LEVELS = {
        'CRITICAL': 5,  # Immediate danger
        'HIGH': 4,      # Serious threat
        'MEDIUM': 3,    # Potential threat
        'LOW': 2,       # Minor concern
        'NONE': 1       # No threat
    }
    
    # Threat scoring for different objects
    OBJECT_THREAT_SCORES = {
        # Weapons
        'knife': 100,
        'gun': 100,
        'rifle': 100,
        'pistol': 100,
        
        # Potential weapons
        'scissors': 40,
        'baseball bat': 30,
        'bottle': 20,
        'hammer': 40,
        
        # Suspicious items
        'backpack': 15,  # Higher if abandoned
        'suitcase': 15,
        'handbag': 10,
        'box': 10,
        
        # Vehicles (context-dependent)
        'car': 5,
        'truck': 10,
        'motorcycle': 5,
        'van': 15,
        
        # People (base score, modified by behavior)
        'person': 5
    }
    
    def __init__(self):
        self.alert_history = []
        self.cooldown_tracker = {}  # Prevent alert spam
        self.dashboard_endpoint = os.getenv('DASHBOARD_ENDPOINT', 'http://localhost:8000/api/alerts')
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detection_metadata": ("DETECTION_METADATA",),
                "track_metadata": ("TRACK_METADATA",),
                "alert_threshold": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 5.0,
                    "tooltip": "Minimum threat score to trigger alert"
                }),
                "cooldown_minutes": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 60,
                    "tooltip": "Minutes before similar alert can trigger again"
                }),
            },
            "optional": {
                "zones_config": ("STRING", {
                    "default": "",
                    "tooltip": "Path to restricted zones configuration JSON"
                }),
                "watchlist_db": ("STRING", {
                    "default": "",
                    "tooltip": "Path to persons of interest database"
                }),
                "send_to_dashboard": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Send alerts to dashboard server"
                }),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "ALERT_DATA", "STRING")
    RETURN_NAMES = ("alert_triggered", "alert_data", "threat_analysis")
    FUNCTION = "analyze_and_alert"
    CATEGORY = "Surveillance/Alert"
    
    def analyze_and_alert(self, detection_metadata: Dict, track_metadata: Dict,
                         alert_threshold: float, cooldown_minutes: int,
                         zones_config: str = "", watchlist_db: str = "",
                         send_to_dashboard: bool = True) -> Tuple[bool, Dict, str]:
        """
        Analyze detections and tracks to generate alerts
        
        Returns:
            alert_triggered: Whether an alert was triggered
            alert_data: Detailed alert information
            threat_analysis: JSON string with threat analysis
        """
        
        # Initialize threat assessment
        threat_assessment = {
            'timestamp': datetime.now().isoformat(),
            'session_id': detection_metadata['session_id'],
            'threats_detected': [],
            'threat_score': 0.0,
            'threat_level': 'NONE',
            'detailed_analysis': {},
            'evidence': {}
        }
        
        # Load configurations
        restricted_zones = self._load_zones_config(zones_config)
        watchlist = self._load_watchlist(watchlist_db)
        
        # 1. Analyze detected objects for weapons/threats
        weapon_analysis = self._analyze_weapons(detection_metadata)
        if weapon_analysis['threat_score'] > 0:
            threat_assessment['threats_detected'].append('weapon_detected')
            threat_assessment['detailed_analysis']['weapons'] = weapon_analysis
            threat_assessment['threat_score'] += weapon_analysis['threat_score']
        
        # 2. Analyze suspicious objects
        suspicious_objects = self._analyze_suspicious_objects(detection_metadata)
        if suspicious_objects['threat_score'] > 0:
            threat_assessment['threats_detected'].append('suspicious_objects')
            threat_assessment['detailed_analysis']['suspicious_objects'] = suspicious_objects
            threat_assessment['threat_score'] += suspicious_objects['threat_score']
        
        # 3. Analyze tracking patterns for suspicious behavior
        behavior_analysis = self._analyze_behavior_patterns(track_metadata)
        if behavior_analysis['threat_score'] > 0:
            threat_assessment['threats_detected'].extend(behavior_analysis['patterns_detected'])
            threat_assessment['detailed_analysis']['behavior'] = behavior_analysis
            threat_assessment['threat_score'] += behavior_analysis['threat_score']
        
        # 4. Check for abandoned objects
        abandoned_analysis = self._check_abandoned_objects(track_metadata)
        if abandoned_analysis['threat_score'] > 0:
            threat_assessment['threats_detected'].append('abandoned_object')
            threat_assessment['detailed_analysis']['abandoned_objects'] = abandoned_analysis
            threat_assessment['threat_score'] += abandoned_analysis['threat_score']
        
        # 5. Zone intrusion detection
        if restricted_zones:
            zone_analysis = self._check_zone_intrusions(track_metadata, restricted_zones)
            if zone_analysis['threat_score'] > 0:
                threat_assessment['threats_detected'].append('zone_intrusion')
                threat_assessment['detailed_analysis']['zone_intrusions'] = zone_analysis
                threat_assessment['threat_score'] += zone_analysis['threat_score']
        
        # 6. Watchlist matching
        if watchlist:
            watchlist_analysis = self._check_watchlist(detection_metadata, watchlist)
            if watchlist_analysis['threat_score'] > 0:
                threat_assessment['threats_detected'].append('watchlist_match')
                threat_assessment['detailed_analysis']['watchlist'] = watchlist_analysis
                threat_assessment['threat_score'] += watchlist_analysis['threat_score']
        
        # 7. Crowd density analysis
        crowd_analysis = self._analyze_crowd_density(detection_metadata)
        if crowd_analysis['risk_level'] in ['high', 'critical']:
            threat_assessment['threats_detected'].append('crowd_risk')
            threat_assessment['detailed_analysis']['crowd'] = crowd_analysis
            threat_assessment['threat_score'] += crowd_analysis['threat_score']
        
        # Calculate final threat level
        threat_assessment['threat_level'] = self._calculate_threat_level(threat_assessment['threat_score'])
        
        # Check if alert should be triggered
        alert_triggered = threat_assessment['threat_score'] >= alert_threshold
        alert_key = self._generate_alert_key(threat_assessment)
        
        # Check cooldown
        if alert_triggered and self._is_in_cooldown(alert_key, cooldown_minutes):
            print(f"Alert suppressed due to cooldown: {alert_key}")
            alert_triggered = False
        
        # Create alert data
        alert_data = {}
        if alert_triggered:
            alert_data = self._create_alert(threat_assessment, detection_metadata, track_metadata)
            
            # Send to dashboard
            if send_to_dashboard:
                self._send_alert_to_dashboard(alert_data)
            
            # Update cooldown tracker
            self.cooldown_tracker[alert_key] = datetime.now()
            
            # Save alert
            self._save_alert(alert_data, detection_metadata)
            
            print(f"⚠️ ALERT TRIGGERED: {threat_assessment['threat_level']} - Score: {threat_assessment['threat_score']:.1f}")
            print(f"   Threats: {', '.join(threat_assessment['threats_detected'])}")
        
        # Return results
        return (
            alert_triggered,
            alert_data,
            json.dumps(threat_assessment, indent=2)
        )
    
    def _analyze_weapons(self, detection_metadata: Dict) -> Dict:
        """Analyze detections for weapons"""
        analysis = {
            'weapons_detected': [],
            'threat_score': 0,
            'frame_evidence': []
        }
        
        weapon_keywords = ['knife', 'gun', 'rifle', 'pistol', 'sword', 'weapon']
        
        for det in detection_metadata.get('all_detections', []):
            class_name = det['class_name'].lower()
            
            # Check if it's a weapon
            for keyword in weapon_keywords:
                if keyword in class_name:
                    analysis['weapons_detected'].append({
                        'type': class_name,
                        'confidence': det['confidence'],
                        'frame_id': det['frame_id'],
                        'timestamp': det['timestamp']
                    })
                    analysis['threat_score'] += self.OBJECT_THREAT_SCORES.get(class_name, 50)
                    analysis['frame_evidence'].append(det['frame_id'])
                    break
        
        # Deduplicate frame evidence
        analysis['frame_evidence'] = list(set(analysis['frame_evidence']))
        
        return analysis
    
    def _analyze_suspicious_objects(self, detection_metadata: Dict) -> Dict:
        """Analyze for suspicious objects"""
        analysis = {
            'suspicious_objects': [],
            'threat_score': 0,
            'categories': defaultdict(int)
        }
        
        suspicious_keywords = ['backpack', 'suitcase', 'bag', 'box', 'package']
        
        for det in detection_metadata.get('all_detections', []):
            class_name = det['class_name'].lower()
            
            for keyword in suspicious_keywords:
                if keyword in class_name:
                    analysis['suspicious_objects'].append({
                        'type': class_name,
                        'confidence': det['confidence'],
                        'frame_id': det['frame_id']
                    })
                    analysis['categories'][class_name] += 1
                    analysis['threat_score'] += self.OBJECT_THREAT_SCORES.get(class_name, 10)
                    break
        
        # Increase score if multiple suspicious objects
        if len(analysis['categories']) > 3:
            analysis['threat_score'] *= 1.5
        
        return analysis
    
    def _analyze_behavior_patterns(self, track_metadata: Dict) -> Dict:
        """Analyze tracking data for suspicious behavior"""
        analysis = {
            'patterns_detected': [],
            'threat_score': 0,
            'suspicious_tracks': []
        }
        
        suspicious_patterns = track_metadata.get('suspicious_analysis', {})
        
        # Loitering
        if suspicious_patterns.get('loitering'):
            for loiterer in suspicious_patterns['loitering']:
                if loiterer['severity'] in ['medium', 'high']:
                    analysis['patterns_detected'].append('loitering')
                    analysis['threat_score'] += 20 if loiterer['severity'] == 'medium' else 40
                    analysis['suspicious_tracks'].append(loiterer['track_id'])
        
        # Erratic movement
        if suspicious_patterns.get('erratic_movement'):
            for track in suspicious_patterns['erratic_movement']:
                if track['severity'] in ['medium', 'high']:
                    analysis['patterns_detected'].append('erratic_movement')
                    analysis['threat_score'] += 15 if track['severity'] == 'medium' else 30
                    analysis['suspicious_tracks'].append(track['track_id'])
        
        # Converging tracks (potential group threat)
        tracks = track_metadata.get('all_tracks', [])
        converging = self._detect_converging_tracks(tracks)
        if converging:
            analysis['patterns_detected'].append('converging_movement')
            analysis['threat_score'] += len(converging) * 10
            analysis['converging_groups'] = converging
        
        return analysis
    
    def _check_abandoned_objects(self, track_metadata: Dict) -> Dict:
        """Check for abandoned objects"""
        analysis = {
            'abandoned_objects': [],
            'threat_score': 0
        }
        
        suspicious_patterns = track_metadata.get('suspicious_analysis', {})
        abandoned = suspicious_patterns.get('abandoned_object', [])
        
        for obj in abandoned:
            analysis['abandoned_objects'].append(obj)
            # High threat score for abandoned bags/packages
            if obj['object_type'] in ['backpack', 'suitcase', 'bag', 'package']:
                analysis['threat_score'] += 50
            else:
                analysis['threat_score'] += 20
        
        return analysis
    
    def _check_zone_intrusions(self, track_metadata: Dict, zones: List[Dict]) -> Dict:
        """Check for intrusions into restricted zones"""
        analysis = {
            'intrusions': [],
            'threat_score': 0
        }
        
        # Implementation depends on zone configuration format
        # This is a placeholder for zone checking logic
        
        return analysis
    
    def _check_watchlist(self, detection_metadata: Dict, watchlist: List[Dict]) -> Dict:
        """Check for persons of interest from watchlist"""
        analysis = {
            'matches': [],
            'threat_score': 0
        }
        
        # Implementation depends on face recognition integration
        # This is a placeholder for watchlist checking
        
        return analysis
    
    def _analyze_crowd_density(self, detection_metadata: Dict) -> Dict:
        """Analyze crowd density for safety risks"""
        analysis = {
            'person_count': 0,
            'density': 'low',
            'risk_level': 'low',
            'threat_score': 0
        }
        
        # Count persons in each frame
        frame_person_counts = defaultdict(int)
        for det in detection_metadata.get('all_detections', []):
            if det['class_name'] == 'person':
                frame_person_counts[det['frame_id']] += 1
        
        if frame_person_counts:
            max_persons = max(frame_person_counts.values())
            avg_persons = np.mean(list(frame_person_counts.values()))
            
            analysis['person_count'] = int(max_persons)
            analysis['avg_person_count'] = float(avg_persons)
            
            # Determine density and risk
            if max_persons > 50:
                analysis['density'] = 'critical'
                analysis['risk_level'] = 'critical'
                analysis['threat_score'] = 40
            elif max_persons > 30:
                analysis['density'] = 'high'
                analysis['risk_level'] = 'high'
                analysis['threat_score'] = 25
            elif max_persons > 15:
                analysis['density'] = 'medium'
                analysis['risk_level'] = 'medium'
                analysis['threat_score'] = 10
            else:
                analysis['density'] = 'low'
                analysis['risk_level'] = 'low'
                analysis['threat_score'] = 0
        
        return analysis
    
    def _detect_converging_tracks(self, tracks: List[Dict]) -> List[List[int]]:
        """Detect groups of tracks converging"""
        converging_groups = []
        
        # Simple proximity-based grouping
        # More sophisticated analysis would consider trajectories
        
        return converging_groups
    
    def _calculate_threat_level(self, threat_score: float) -> str:
        """Calculate threat level from score"""
        if threat_score >= 80:
            return 'CRITICAL'
        elif threat_score >= 60:
            return 'HIGH'
        elif threat_score >= 40:
            return 'MEDIUM'
        elif threat_score >= 20:
            return 'LOW'
        else:
            return 'NONE'
    
    def _generate_alert_key(self, threat_assessment: Dict) -> str:
        """Generate unique key for alert cooldown tracking"""
        threats = '_'.join(sorted(threat_assessment['threats_detected']))
        return f"{threat_assessment['session_id']}_{threats}"
    
    def _is_in_cooldown(self, alert_key: str, cooldown_minutes: int) -> bool:
        """Check if alert is in cooldown period"""
        if alert_key not in self.cooldown_tracker:
            return False
        
        last_alert_time = self.cooldown_tracker[alert_key]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.now() - last_alert_time < cooldown_period
    
    def _create_alert(self, threat_assessment: Dict, detection_metadata: Dict, 
                     track_metadata: Dict) -> Dict:
        """Create detailed alert data"""
        
        # Gather frame evidence
        frame_evidence = set()
        for analysis in threat_assessment['detailed_analysis'].values():
            if 'frame_evidence' in analysis:
                frame_evidence.update(analysis['frame_evidence'])
        
        # Gather track evidence
        track_evidence = set()
        for analysis in threat_assessment['detailed_analysis'].values():
            if 'suspicious_tracks' in analysis:
                track_evidence.update(analysis['suspicious_tracks'])
        
        alert = {
            'alert_id': str(uuid.uuid4()),
            'session_id': threat_assessment['session_id'],
            'timestamp': datetime.now().isoformat(),
            'threat_level': threat_assessment['threat_level'],
            'threat_score': threat_assessment['threat_score'],
            'alert_type': 'multi_threat' if len(threat_assessment['threats_detected']) > 1 else threat_assessment['threats_detected'][0] if threat_assessment['threats_detected'] else 'unknown',
            'threats_detected': threat_assessment['threats_detected'],
            'description': self._generate_alert_description(threat_assessment),
            'evidence': {
                'frame_ids': list(frame_evidence),
                'track_ids': list(track_evidence),
                'detailed_analysis': threat_assessment['detailed_analysis']
            },
            'location': detection_metadata.get('video_source', 'Unknown'),
            'requires_action': threat_assessment['threat_level'] in ['HIGH', 'CRITICAL'],
            'recommended_actions': self._generate_recommendations(threat_assessment),
            'metadata': {
                'total_frames_analyzed': detection_metadata.get('total_frames_processed', 0),
                'total_detections': len(detection_metadata.get('all_detections', [])),
                'total_tracks': track_metadata.get('total_tracks', 0)
            }
        }
        
        return alert
    
    def _generate_alert_description(self, threat_assessment: Dict) -> str:
        """Generate human-readable alert description"""
        threats = threat_assessment['threats_detected']
        level = threat_assessment['threat_level']
        
        if 'weapon_detected' in threats:
            return f"{level} ALERT: Weapon detected in surveillance footage"
        elif 'abandoned_object' in threats:
            return f"{level} ALERT: Abandoned object detected - potential security threat"
        elif 'loitering' in threats:
            return f"{level} ALERT: Suspicious loitering behavior detected"
        elif 'zone_intrusion' in threats:
            return f"{level} ALERT: Unauthorized access to restricted zone"
        elif 'watchlist_match' in threats:
            return f"{level} ALERT: Person of interest detected from watchlist"
        elif 'crowd_risk' in threats:
            return f"{level} ALERT: Dangerous crowd density detected"
        else:
            return f"{level} ALERT: Multiple security concerns detected"
    
    def _generate_recommendations(self, threat_assessment: Dict) -> List[str]:
        """Generate recommended actions based on threats"""
        recommendations = []
        level = threat_assessment['threat_level']
        threats = threat_assessment['threats_detected']
        
        if level == 'CRITICAL':
            recommendations.append("IMMEDIATE: Contact law enforcement")
            recommendations.append("IMMEDIATE: Dispatch security to location")
            recommendations.append("IMMEDIATE: Initiate lockdown procedures if applicable")
        elif level == 'HIGH':
            recommendations.append("URGENT: Send security patrol to investigate")
            recommendations.append("URGENT: Review live feeds from area")
            recommendations.append("Monitor situation closely for escalation")
        
        # Specific recommendations
        if 'weapon_detected' in threats:
            recommendations.append("Identify and track armed individual")
            recommendations.append("Prepare for potential evacuation")
        
        if 'abandoned_object' in threats:
            recommendations.append("Cordon off area around suspicious object")
            recommendations.append("Consider bomb squad if object remains unidentified")
        
        if 'loitering' in threats:
            recommendations.append("Dispatch security to question individual")
            recommendations.append("Check for associated vehicles or accomplices")
        
        if 'zone_intrusion' in threats:
            recommendations.append("Verify access credentials")
            recommendations.append("Review access logs for anomalies")
        
        if 'crowd_risk' in threats:
            recommendations.append("Deploy crowd control measures")
            recommendations.append("Prepare medical teams for potential incidents")
        
        return recommendations
    
    def _send_alert_to_dashboard(self, alert_data: Dict):
        """Send alert to dashboard server"""
        try:
            response = requests.post(
                self.dashboard_endpoint,
                json=alert_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            if response.status_code == 200:
                print(f"✓ Alert sent to dashboard: {alert_data['alert_id']}")
            else:
                print(f"⚠️ Dashboard response: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Failed to send alert to dashboard: {e}")
    
    def _save_alert(self, alert_data: Dict, detection_metadata: Dict):
        """Save alert to file system"""
        # Get storage directory from detection metadata
        if 'storage_directory' in detection_metadata:
            storage_dir = Path(detection_metadata['storage_directory'])
        else:
            storage_dir = Path(f"./surveillance_storage/alerts/{alert_data['session_id']}")
        
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Save alert
        alert_file = storage_dir / f"alert_{alert_data['alert_id']}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        print(f"✓ Alert saved: {alert_file}")
    
    def _load_zones_config(self, zones_config_path: str) -> List[Dict]:
        """Load restricted zones configuration"""
        if zones_config_path and os.path.exists(zones_config_path):
            with open(zones_config_path, 'r') as f:
                return json.load(f)
        return []
    
    def _load_watchlist(self, watchlist_path: str) -> List[Dict]:
        """Load watchlist database"""
        if watchlist_path and os.path.exists(watchlist_path):
            with open(watchlist_path, 'r') as f:
                return json.load(f)
        return []

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AlertNode": AlertNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AlertNode": "Security Alert System"
}