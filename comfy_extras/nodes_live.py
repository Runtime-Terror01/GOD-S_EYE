import nodes
import torch
from typing import Tuple

from api_server.services.live_processing import get_live_manager


class LiveFrameInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "wait_for_frame": ("BOOLEAN", {"default": True, "tooltip": "Block until a new frame arrives"}),
                "timeout_ms": ("INT", {"default": 0, "min": 0, "max": 60000, "step": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "read"
    CATEGORY = "live"

    def read(self, wait_for_frame: bool, timeout_ms: int) -> Tuple[torch.Tensor]:
        # The server singleton is initialized in PromptServer; we access the global manager
        mgr = get_live_manager()
        if wait_for_frame:
            timeout = None if timeout_ms <= 0 else float(timeout_ms) / 1000.0
            mgr.frame_buffer.wait_for_update(timeout)
        frame = mgr.frame_buffer.get_frame()
        if frame is None:
            # Provide a 1x1 black pixel if no frame yet
            frame = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        return (frame,)


NODE_CLASS_MAPPINGS = {
    "LiveFrameInput": LiveFrameInput,
}



