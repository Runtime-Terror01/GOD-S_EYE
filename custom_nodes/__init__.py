"""
ComfyUI Threat Detection Node Suite
RF-DETR based threat detection system for ComfyUI

Dependencies:
pip install rfdetr==1.2.1 supervision==0.26.1
"""

from .load_model import LoadThreatDetectionModel
from .threat_detection import ThreatDetectionNode
from .frame_extractor import VideoFrameExtractor
from .alert import AlertNode
from .vlm_report import VLMReportNode
from .preview import PreviewNode
from .storage_cleanup import StorageCleanupNode

NODE_CLASS_MAPPINGS = {
    "LoadThreatDetectionModel": LoadThreatDetectionModel,
    "ThreatDetectionNode": ThreatDetectionNode,
    "VideoFrameExtractor": VideoFrameExtractor,
    "AlertNode": AlertNode,
    "VLMReportNode": VLMReportNode,
    "PreviewNode": PreviewNode,
    "StorageCleanupNode": StorageCleanupNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadThreatDetectionModel": "Load Threat Detection Model",
    "ThreatDetectionNode": "Threat Detection (RF-DETR)",
    "VideoFrameExtractor": "Video Frame Extractor",
    "AlertNode": "Threat Alert Analysis",
    "VLMReportNode": "VLM Security Report",
    "PreviewNode": "Text Preview",
    "StorageCleanupNode": "Storage Cleanup",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']