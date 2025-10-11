"""
Preview Node
Display text content in ComfyUI interface
"""

class PreviewNode:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "preview_text"
    CATEGORY = "Threat Detection"
    OUTPUT_NODE = True
    
    def preview_text(self, text):
        """Display text content"""
        print(f"\n=== PREVIEW ===")
        print(text)
        print("=== END PREVIEW ===\n")
        
        return {"ui": {"text": [text]}}

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PreviewNode": PreviewNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewNode": "Text Preview"
}