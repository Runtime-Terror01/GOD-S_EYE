"""
ComfyUI Surveillance Nodes Package
Provides video surveillance processing nodes for ComfyUI
"""

import os
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import node classes
from .nodes.frame_extractor_node import FrameExtractorNode
from .nodes.object_detection_node import ObjectDetectionNode
from .nodes.tracking_node import TrackingNode
# from .nodes.vlm_node import VLMNode
from .nodes.alert_node import AlertNode
from .nodes.vlm_analysis_node import VLMAnalysisNode as VLMNode

# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    "SurveillanceFrameExtractor": FrameExtractorNode,
    "SurveillanceObjectDetection": ObjectDetectionNode,
    "SurveillanceTracking": TrackingNode,
    "SurveillanceVLM": VLMNode,
    "SurveillanceAlert": AlertNode,
    # "SurveillanceVLMAnalysis": VLMAnalysisNode
}

# Node Display Names in ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SurveillanceFrameExtractor": "Frame Extractor (Surveillance)",
    "SurveillanceObjectDetection": "Object Detection (Surveillance)",
    "SurveillanceTracking": "Object Tracking (Surveillance)",
    "SurveillanceVLM": "VLM Analysis (Surveillance)",
    "SurveillanceAlert": "Alert Generator (Surveillance)",
    # "SurveillanceVLMAnalysis": "VLM Scene Analysis (Surveillance)"
}

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("🔒 Surveillance Nodes Loaded:")
print(f"   - Frame Extractor")
print(f"   - Object Detection (YOLOv8)")
print(f"   - Object Tracking (ByteTrack)")
print(f"   - VLM Analysis (Gemini)")
print(f"   - Alert Generator")
# print(f"   - VLM Scene Analysis & Reporting")
