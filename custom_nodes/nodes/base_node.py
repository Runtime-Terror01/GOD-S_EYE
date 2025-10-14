"""
Base Node Abstract Class
All surveillance nodes inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from pathlib import Path
import json


class BaseNode(ABC):
    """
    Abstract base class for all surveillance pipeline nodes.
    Provides common interface and metadata management.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the node with configuration
        
        Args:
            config: Dictionary containing node configuration
        """
        self.config = config or {}
        self.storage_path = None
        self.metadata = {}
        
    @abstractmethod
    def process(self, *inputs) -> Tuple:
        """
        Process inputs and return standardized outputs
        
        This method must be implemented by all subclasses.
        Input and output types should follow the standardized data types.
        
        Returns:
            tuple: Standardized outputs as defined in data_types.py
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def save_metadata(self, storage_path: str) -> None:
        """
        Write metadata files to disk following surveillance_storage layout
        
        Args:
            storage_path: Path to session directory (e.g., surveillance_storage/session_<id>/)
        """
        if not self.metadata:
            return
        
        self.storage_path = Path(storage_path)
        metadata_dir = self.storage_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Write metadata based on node type
        self._write_metadata(metadata_dir)
    
    def _write_metadata(self, metadata_dir: Path) -> None:
        """
        Internal method to write specific metadata files.
        Override in subclasses to write specific metadata files.
        
        Args:
            metadata_dir: Path to metadata directory
        """
        pass
    
    def _save_json(self, filepath: Path, data: Dict[str, Any]) -> None:
        """
        Helper method to save JSON data
        
        Args:
            filepath: Full path to JSON file
            data: Dictionary to save as JSON
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """
        Helper method to load JSON data
        
        Args:
            filepath: Full path to JSON file
            
        Returns:
            Dictionary loaded from JSON
        """
        if not filepath.exists():
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
