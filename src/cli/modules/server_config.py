"""
ServerConfig - Centralized configuration management for OpenArc.

This module handles all configuration file operations without any CLI/presentation logic.
"""
import logging
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, cast

import yaml

from ..utils import get_config_file_path

logger = logging.getLogger(__name__)

# Matches ${VAR} and ${VAR:-default} inside a string scalar.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ServerConfig:
    """Manages OpenArc server and model configurations."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize ServerConfig with a config file path.

        Args:
            config_file: Path to the config file. If None, OPENARC_CONFIG_FILE env var when set,
            otherwise defaults to config.yaml in project root.
        """
        self.config_file = get_config_file_path() if config_file is None else config_file

    @staticmethod
    def _interpolate_env(value: Any, path: str = "") -> Any:
        """Recursively expand ${VAR} / ${VAR:-default} in string scalars.

        Interpolation is applied to values only, never to mapping keys. An
        unset variable with no default expands to an empty string and logs a
        warning rather than raising, so a config referencing an unset
        OPENARC_AUTOLOAD_MODELS still loads.
        """
        if isinstance(value, str):
            def _replace(match: re.Match) -> str:
                name, default = match.group(1), match.group(2)
                if name in os.environ:
                    return os.environ[name]
                if default is not None:
                    return default
                logger.warning(
                    f"Config: ${{{name}}} at '{path or '<root>'}' is unset; using empty string"
                )
                return ""

            return _ENV_PATTERN.sub(_replace, value)
        if isinstance(value, dict):
            return {
                key: ServerConfig._interpolate_env(
                    val, f"{path}.{key}" if path else str(key)
                )
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [
                ServerConfig._interpolate_env(item, f"{path}[{idx}]")
                for idx, item in enumerate(value)
            ]
        return value

    def load_config(self) -> Dict[str, Any]:
        """
        Load full configuration from the YAML config file.

        ${VAR} references in string scalars are resolved from the environment
        after parsing.

        Returns:
            Configuration dictionary, or empty dict if file doesn't exist or is invalid.
        """
        if not self.config_file.exists():
            return {}

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except (yaml.YAMLError, FileNotFoundError, OSError) as exc:
            logger.error(f"Config: failed to read {self.config_file}: {exc}")
            return {}

        if not config:
            return {}
        if not isinstance(config, dict):
            logger.error(
                f"Config: {self.config_file} must contain a mapping at the top level, "
                f"got {type(config).__name__}"
            )
            return {}

        return cast(Dict[str, Any], self._interpolate_env(config))

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration to the YAML config file. Does nothing if an on-disk
        config file exists and its parsed contents are identical to the provided
        config, allowing the config to live on a read-only filesystem.

        Comparison is done on the parsed structure rather than raw text, since
        YAML round-tripping legitimately reformats the document.

        Args:
            config: Configuration dictionary to save.
        """

        if self.config_file.exists():
            if self.load_config() == self._interpolate_env(config):
                return  # No changes, skip writing

        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    
    def save_server_config(self, host: str, port: int) -> Path:
        """
        Record the server bind address.

        Deliberately a no-op on disk. config.yaml is hand-authored and may
        contain comments and ${VAR} references that a YAML round-trip cannot
        preserve, so `openarc serve start` must not rewrite it. host/port are
        supplied on the command line and read back from there.

        Args:
            host: Server host address.
            port: Server port number.

        Returns:
            Path to the config file.
        """
        return self.config_file
    
    def load_server_config(self) -> Dict[str, Any]:
        """
        Load server configuration.
        
        Returns:
            Server configuration dict with 'host' and 'port' keys.
            Returns defaults if not configured.
        """
        config = self.load_config()
        server = config.get("server") if config else None
        if isinstance(server, dict):
            # Blank values (e.g. a bare `port:` key) parse as None; fall back to
            # the default rather than handing a None back to callers.
            host = server.get("host") or "localhost"
            port = server.get("port") or 8000
            return {"host": host, "port": port}

        return {"host": "localhost", "port": 8000}
    
    def save_model_config(self, model_name: str, model_config: Dict[str, Any]) -> None:
        """
        Save model configuration under models.<model_name>.load_config.

        Writing into load_config keeps every model entry in the canonical
        nested shape, whether it was authored by this command or by hand.

        Args:
            model_name: Name of the model.
            model_config: Model load configuration dictionary.
        """
        config = self.load_config()

        if "models" not in config:
            config["models"] = {}

        config["models"][model_name] = {"load_config": model_config}
        self.save_config(config)

    def save_model_entry(self, model_name: str, entry: Dict[str, Any]) -> None:
        """Save a full models.<model_name> entry, preserving the nested shape.

        Unlike save_model_config, this stores sibling blocks (sampler_config,
        *_config) alongside load_config rather than only load fields.
        """
        config = self.load_config()
        if "models" not in config:
            config["models"] = {}
        config["models"][model_name] = entry
        self.save_config(config)

    @staticmethod
    def _split_model_entry(
        model_config: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Split a models.<name> entry into (load fields, config blocks).

        The canonical shape nests load fields under `load_config` and keeps the
        request-default blocks (sampler_config, runtime_config,
        scheduler_config, *_config) as siblings. Returns a tuple so callers can
        route each half to the right place.
        """
        from src.server.schemas.modeling.config_blocks import BLOCK_CONTRACTS

        if not isinstance(model_config, dict):
            return {}, {}

        nested = model_config.get("load_config")
        if not isinstance(nested, dict):
            # Flat-form entry: the whole thing is load fields.
            return dict(model_config), {}

        load_fields = dict(nested)
        blocks: Dict[str, Any] = {}
        for key, value in model_config.items():
            if key == "load_config":
                continue
            if key in BLOCK_CONTRACTS:
                blocks[key] = value
            else:
                # runtime_config / scheduler_config and other load-time siblings.
                load_fields[key] = value
        return load_fields, blocks

    @staticmethod
    def _unwrap_model_entry(model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a nested models.<name> entry into a single load-config dict."""
        load_fields, _ = ServerConfig._split_model_entry(model_config)
        return load_fields

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get model configuration by name.

        Args:
            model_name: Name of the model.

        Returns:
            Flattened model configuration dict, or None if not found. Relative paths are
            resolved against the config file's directory, allowing for configs to be
            packaged with models.
        """
        config = self.load_config()
        models = cast(Dict[str, Dict[str, Any]], config.get("models", {}))
        model = models.get(model_name)
        if not model:
            return None
        return self._resolve_model_paths(self._unwrap_model_entry(model))

    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all model configurations.

        Returns:
            Dictionary mapping model names to their flattened configurations. Relative
            paths are resolved against the config file's directory, allowing for configs
            to be packaged with models.
        """
        config = self.load_config()
        models = cast(Dict[str, Dict[str, Any]], config.get("models", {}))
        return {
            name: self._resolve_model_paths(self._unwrap_model_entry(cfg))
            for name, cfg in models.items()
        }

    def _resolve_model_paths(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of model_config with relative model_path, draft_model_path,
        and cache_dir made absolute by joining them onto the config file's directory."""
        resolved = dict(model_config)

        path = resolved.get("model_path")
        if path and not Path(path).is_absolute():
            resolved["model_path"] = str((self.config_file.parent / path).resolve())

        draft_model_path = resolved.get("draft_model_path")
        if draft_model_path and not Path(draft_model_path).is_absolute():
            resolved["draft_model_path"] = str((self.config_file.parent / draft_model_path).resolve())

        cache_dir = resolved.get("cache_dir")
        if cache_dir and not Path(cache_dir).is_absolute():
            resolved["cache_dir"] = str((self.config_file.parent / cache_dir).resolve())

        return resolved

    def get_model_load_config(self, model_name: str) -> Optional["ModelLoadConfig"]:
        """Build a validated ModelLoadConfig for a named model in config.yaml.

        model_name is injected from the mapping key, overriding anything the
        entry itself declares, so the key is authoritative.

        Returns:
            A validated ModelLoadConfig, or None if the model is not configured.

        Raises:
            ValueError: If the entry is malformed or carries config blocks that
                do not match its model_type.
        """
        from pydantic import ValidationError

        from src.server.schemas.registration import ModelLoadConfig

        config = self.load_config()
        models = cast(Dict[str, Dict[str, Any]], config.get("models", {}))
        entry = models.get(model_name)
        if not entry:
            return None

        flattened, blocks = self._split_model_entry(entry)
        flattened = self._resolve_model_paths(flattened)
        flattened["model_name"] = model_name
        if blocks:
            flattened["model_config_blocks"] = blocks

        try:
            load_config = ModelLoadConfig(**flattened)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid configuration for model '{model_name}': {exc}"
            ) from exc

        load_config.validate_config_blocks()
        return load_config

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

