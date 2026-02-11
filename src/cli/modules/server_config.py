"""
ServerConfig - Centralized configuration management for OpenArc.

This module handles all configuration file operations without any CLI/presentation logic.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, List


class ServerConfig:
    """Manages OpenArc server and model configurations."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize ServerConfig with a config file path.
        
        Args:
            config_file: Path to the config file. If None, defaults to openarc_config.json in project root.
        """
        if config_file is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_file = project_root / "openarc_config.json"
        
        self.config_file = Path(config_file)
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load full configuration from JSON config file.
        
        Returns:
            Configuration dictionary, or empty dict if file doesn't exist or is invalid.
        """
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
                return config if config else {}
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration to JSON config file.
        
        Args:
            config: Configuration dictionary to save.
        """
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    def save_server_config(self, host: str, port: int) -> Path:
        """
        Save server configuration.
        
        Args:
            host: Server host address.
            port: Server port number.
            
        Returns:
            Path to the saved config file.
        """
        config = self.load_config()
        config.update({
            "server": {
                "host": host,
                "port": port
            },
            "created_by": "openarc-cli",
            "version": "1.0"
        })
        
        self.save_config(config)
        return self.config_file
    
    def load_server_config(self) -> Dict[str, Any]:
        """
        Load server configuration.
        
        Returns:
            Server configuration dict with 'host' and 'port' keys.
            Returns defaults if not configured.
        """
        config = self.load_config()
        if config and "server" in config:
            return config["server"]
        
        return {"host": "localhost", "port": 8000}
    
    def save_model_config(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """
        Save model configuration.
        
        Args:
            model_name: Name of the model.
            model_config: Model configuration dictionary.
        """
        config = self.load_config()
        
        if "models" not in config:
            config["models"] = {}
        
        config["models"][model_name] = model_config
        self.save_config(config)
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get model configuration by name.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Model configuration dict, or None if not found.
        """
        config = self.load_config()
        models = config.get("models", {})
        return models.get(model_name)
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all model configurations.
        
        Returns:
            Dictionary mapping model names to their configurations.
        """
        config = self.load_config()
        return config.get("models", {})
    
    def remove_model_config(self, model_name: str) -> bool:
        """
        Remove model configuration by name.
        
        Args:
            model_name: Name of the model to remove.
            
        Returns:
            True if model was removed, False if model was not found.
        """
        config = self.load_config()
        models = config.get("models", {})
        
        if model_name not in models:
            return False
        
        del models[model_name]
        config["models"] = models
        self.save_config(config)
        
        return True
    
    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model configuration exists.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            True if model exists, False otherwise.
        """
        return self.get_model_config(model_name) is not None
    
    def get_model_names(self) -> List[str]:
        """
        Get list of all configured model names.
        
        Returns:
            List of model names.
        """
        return list(self.get_all_models().keys())
    
    def get_base_url(self) -> str:
        """
        Get the base URL for the OpenArc server.
        
        Returns:
            Base URL string (e.g., 'http://localhost:8000').
        """
        server_config = self.load_server_config()
        return f"http://{server_config['host']}:{server_config['port']}"

