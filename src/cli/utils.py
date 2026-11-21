"""
Utility functions for OpenArc CLI.
"""
import json
import os
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterable

TOOL_CALL_TYPES = {
    "MuseGlimmerForConditionalGeneration": "museglimmer",
    "Gemma4ForConditionalGeneration": "gemma4",
    "Gemma4UnifiedForConditionalGeneration": "gemma4",
    "Qwen3_5ForConditionalGeneration": "qwen35",
    "Qwen3_5MoeForConditionalGeneration": "qwen35",
}

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
    then defaulting to config.yaml in the project root.
    """
    env_path = os.environ.get("OPENARC_CONFIG_FILE")
    if env_path:
        return Path(env_path)
    else:
        project_root = Path(__file__).parent.parent.parent
        return project_root / "config.yaml"

def _architecture_values(config: Any) -> Iterable[str]:
    if not isinstance(config, dict):
        return ()

    architectures = config.get("architectures")
    if isinstance(architectures, str):
        return (architectures,)
    if isinstance(architectures, list):
        return tuple(item for item in architectures if isinstance(item, str))
    return ()

def resolve_tool_call_parser(model_path: str) -> str | None:
    config_path = Path(model_path) / "config.json"

    try:
        config_text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None

    try:
        config = json.loads(config_text)
    except JSONDecodeError:
        return None

    for architecture in _architecture_values(config):
        type = TOOL_CALL_TYPES.get(architecture)
        if type is not None:
            return type

    return None
