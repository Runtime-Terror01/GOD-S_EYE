"""
Load Threat Detection Model Node
Downloads and loads RF-DETR checkpoint from Hugging Face
"""

import os
import requests
import torch
from tqdm import tqdm

try:
    from rfdetr import RFDETRNano
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Warning: rfdetr library not available. Install with: pip install rfdetr==1.2.1")

class LoadThreatDetectionModel:
    def __init__(self):
        self.model = None
        self.weights_path = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["Subh775/Threat-Detection-RF-DETR"], {
                    "default": "Subh775/Threat-Detection-RF-DETR"
                }),
                "resolution": ("INT", {
                    "default": 640,
                    "min": 320,
                    "max": 1280,
                    "step": 32,
                    "display": "slider",
                    "tooltip": "Input image resolution for RF-DETR model. Higher = better accuracy but slower processing. Lower = faster but may miss small objects."
                }),
                "force_reload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force reload the model even if it's already cached"
                }),
            },
        }
    
    RETURN_TYPES = ("THREAT_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Threat Detection"
    
    def download_weights(self, model_name):
        """Download model weights from Hugging Face"""
        
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Define weights path
        weights_filename = "checkpoint_best_total.pth"
        weights_path = os.path.join(models_dir, weights_filename)
        
        # Check if weights already exist
        if os.path.exists(weights_path):
            print(f"✓ Model weights found at: {weights_path}")
            return weights_path
        
        # Download weights
        weights_url = f"https://huggingface.co/{model_name}/resolve/main/{weights_filename}"
        
        print(f"Downloading model weights from: {weights_url}")
        print(f"Saving to: {weights_path}")
        
        try:
            response = requests.get(weights_url, stream=True)
            response.raise_for_status()
            
            # Get total file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            with open(weights_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print("✓ Download complete!")
            return weights_path
            
        except Exception as e:
            if os.path.exists(weights_path):
                os.remove(weights_path)
            raise RuntimeError(f"Failed to download model weights: {str(e)}")
    
    def load_model(self, model_name, resolution, force_reload):
        """Load RF-DETR model with threat detection weights"""
        
        if not RFDETR_AVAILABLE:
            raise ImportError(
                "rfdetr library is required but not installed. "
                "Please install it with: pip install rfdetr==1.2.1"
            )
        
        print(f"\n=== Loading Threat Detection Model ===")
        print(f"Model: {model_name}")
        print(f"Resolution: {resolution}x{resolution}")
        print(f"Force reload: {force_reload}")
        
        # Download weights if needed
        weights_path = self.download_weights(model_name)
        
        # Check if we need to reload the model
        if (self.model is None or 
            self.weights_path != weights_path or 
            force_reload):
            
            print("Loading RF-DETR model...")
            
            try:
                # Initialize model
                model = RFDETRNano(
                    resolution=resolution, 
                    pretrain_weights=weights_path
                )
                
                # Optimize for inference
                model.optimize_for_inference()
                
                # Auto-detect device
                if torch.cuda.is_available():
                    device = "cuda"
                    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
                    
                    # Enable optimizations for batch processing
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                else:
                    device = "cpu"
                    print("✓ Using CPU")
                
                # Store model info
                self.model = model
                self.weights_path = weights_path
                
                print("✓ Model loaded successfully!")
                
                # Create model wrapper with metadata
                model_wrapper = {
                    'model': model,
                    'resolution': resolution,
                    'device': device,
                    'weights_path': weights_path,
                    'model_name': model_name,
                    'threat_classes': {
                        1: "Gun",
                        2: "Explosive", 
                        3: "Grenade",
                        4: "Knife"
                    }
                }
                
                return (model_wrapper,)
                
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")
        
        else:
            print("✓ Using cached model")
            
            # Return cached model with updated metadata
            model_wrapper = {
                'model': self.model,
                'resolution': resolution,
                'device': "cuda" if torch.cuda.is_available() else "cpu",
                'weights_path': self.weights_path,
                'model_name': model_name,
                'threat_classes': {
                    1: "Gun",
                    2: "Explosive", 
                    3: "Grenade",
                    4: "Knife"
                }
            }
            
            return (model_wrapper,)