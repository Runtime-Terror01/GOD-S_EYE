from .frame_extractor_node import VideoFrameExtractorNode
from .unified_frame_extractor_node import UnifiedVideoFrameExtractor
from .object_detection_node import ObjectDetectionNode
from .tracking_node import TrackingNode
from .alert_node import AlertNode
from .vlm_report_node import VLMReportNode


NODE_CLASS_MAPPINGS = {
    "VideoFrameExtractorNode": VideoFrameExtractorNode,
    "ObjectDetectionNode": ObjectDetectionNode,
    "TrackingNode": TrackingNode,
    "AlertNode": AlertNode,
    "VLMReportNode": VLMReportNode,
    "UnifiedVideoFrameExtractor": UnifiedVideoFrameExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameExtractorNode": "Extract Video Frames",
    "ObjectDetectionNode": "YOLO Object Detection",
    "TrackingNode": "Object Tracking (ByteTrack)",
    "AlertNode": "Security Alert System",
    "VLMReportNode": "VLM Scene Analysis & Reporting",
    "UnifiedVideoFrameExtractor": "Unified Video Frame Extractor",
}



