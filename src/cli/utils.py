"""
Utility functions for OpenArc CLI.
"""
import os
from pathlib import Path


def validate_model_path(model_path):
    """
    Validate that model_path contains OpenVINO model files.
    Checks for at least one file with "_model.bin" and one file with "_model.xml" in filename.
    Returns True if valid, False otherwise.
    """
    path = Path(model_path)
    
    # Resolve the path
    if not path.exists():
        return False
    
    # Determine search directory - if path is a file, use its parent; if directory, use it
    if path.is_file():
        search_dir = path.parent
    else:
        search_dir = path
        
    if not search_dir.is_absolute():
        # Search relative to the config file directory
        config_file =  get_config_file_path()
        search_dir = (config_file.parent / search_dir).resolve()
    
    # Check for required files
    has_bin = False
    has_xml = False
    
    try:
        for file_path in search_dir.rglob("*"):
            if file_path.is_file():
                filename = file_path.name
                if "_model.bin" in filename:
                    has_bin = True
                if "_model.xml" in filename:
                    has_xml = True
                if has_bin and has_xml:
                    return True
    except (OSError, PermissionError):
        return False
    
    return False

def get_config_file_path():
    """
    Get the path to the config file, checking the OPENARC_CONFIG_FILE environment variable first,
    then defaulting to openarc_config.json in the project root.
    """
    env_path = os.environ.get("OPENARC_CONFIG_FILE")
    if env_path:
        return Path(env_path)
    else:
        project_root = Path(__file__).parent.parent.parent
        return project_root / "openarc_config.json"
