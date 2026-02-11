"""
Utility functions for OpenArc CLI.
"""
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
