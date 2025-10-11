"""
Storage Cleanup Utility
Manages frame storage and cleanup operations
"""

import os
import shutil
import time
from datetime import datetime, timedelta

class StorageCleanupNode:
    def __init__(self):
        self.storage_dir = os.path.join(os.getcwd(), "storage")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cleanup_older_than_hours": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 168,  # 1 week
                    "step": 1,
                    "tooltip": "Delete frame storage older than this many hours"
                }),
                "force_cleanup": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force cleanup regardless of age"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleanup_report",)
    FUNCTION = "cleanup_storage"
    CATEGORY = "Threat Detection"
    
    def get_directory_size(self, directory):
        """Calculate total size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    def format_size(self, size_bytes):
        """Format size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
    
    def cleanup_storage(self, cleanup_older_than_hours, force_cleanup):
        """Clean up old frame storage directories"""
        
        print(f"\n=== Storage Cleanup ===")
        print(f"Storage directory: {self.storage_dir}")
        print(f"Cleanup older than: {cleanup_older_than_hours} hours")
        print(f"Force cleanup: {force_cleanup}")
        
        if not os.path.exists(self.storage_dir):
            return ("Storage directory does not exist.",)
        
        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(hours=cleanup_older_than_hours)
        
        # Scan storage directory
        total_dirs = 0
        cleaned_dirs = 0
        total_size_before = 0
        total_size_cleaned = 0
        cleanup_report = []
        
        cleanup_report.append("=== STORAGE CLEANUP REPORT ===")
        cleanup_report.append(f"Cleanup started: {datetime.now().isoformat()}")
        cleanup_report.append(f"Cutoff time: {cutoff_time.isoformat()}")
        cleanup_report.append("")
        
        try:
            # Get initial storage size
            total_size_before = self.get_directory_size(self.storage_dir)
            cleanup_report.append(f"Total storage size before cleanup: {self.format_size(total_size_before)}")
            cleanup_report.append("")
            
            # Scan subdirectories
            for item in os.listdir(self.storage_dir):
                item_path = os.path.join(self.storage_dir, item)
                
                if os.path.isdir(item_path):
                    total_dirs += 1
                    
                    # Get directory modification time
                    dir_mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                    
                    # Check if directory should be cleaned
                    should_clean = force_cleanup or dir_mtime < cutoff_time
                    
                    if should_clean:
                        try:
                            # Calculate size before deletion
                            dir_size = self.get_directory_size(item_path)
                            
                            # Remove directory
                            shutil.rmtree(item_path)
                            
                            cleaned_dirs += 1
                            total_size_cleaned += dir_size
                            
                            cleanup_report.append(f"✓ Cleaned: {item} ({self.format_size(dir_size)}, modified: {dir_mtime.strftime('%Y-%m-%d %H:%M:%S')})")
                            
                        except Exception as e:
                            cleanup_report.append(f"✗ Failed to clean {item}: {e}")
                    else:
                        cleanup_report.append(f"• Kept: {item} (modified: {dir_mtime.strftime('%Y-%m-%d %H:%M:%S')})")
            
            # Final statistics
            total_size_after = self.get_directory_size(self.storage_dir)
            
            cleanup_report.append("")
            cleanup_report.append("=== CLEANUP SUMMARY ===")
            cleanup_report.append(f"Total directories found: {total_dirs}")
            cleanup_report.append(f"Directories cleaned: {cleaned_dirs}")
            cleanup_report.append(f"Directories kept: {total_dirs - cleaned_dirs}")
            cleanup_report.append(f"Size before cleanup: {self.format_size(total_size_before)}")
            cleanup_report.append(f"Size after cleanup: {self.format_size(total_size_after)}")
            cleanup_report.append(f"Space freed: {self.format_size(total_size_cleaned)}")
            cleanup_report.append(f"Cleanup completed: {datetime.now().isoformat()}")
            
            print(f"✓ Cleanup complete: {cleaned_dirs}/{total_dirs} directories cleaned, {self.format_size(total_size_cleaned)} freed")
            
        except Exception as e:
            error_msg = f"Error during cleanup: {e}"
            cleanup_report.append(error_msg)
            print(f"Error: {error_msg}")
        
        return ("\n".join(cleanup_report),)

# Node class mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "StorageCleanupNode": StorageCleanupNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StorageCleanupNode": "Storage Cleanup"
}