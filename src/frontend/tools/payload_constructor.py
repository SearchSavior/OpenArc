
from pydantic import BaseModel
from typing import Optional
import requests
import os


# Default OpenARC URL
DEFAULT_OPENARC_PORT = 8000
OPENARC_URL = f"http://localhost:{DEFAULT_OPENARC_PORT}"

# Update URL if custom port is provided
def update_openarc_url(openarc_port=DEFAULT_OPENARC_PORT):
    global OPENARC_URL
    OPENARC_URL = f"http://localhost:{openarc_port}"

def get_auth_headers():
    """Get authorization headers with bearer token if available"""
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENARC_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class LoadModelConfig(BaseModel):
    id_model: str
    use_cache: bool
    device: str
    export_model: bool
    is_vision_model: bool = False 
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    dynamic_shapes: bool = True

class OVConfig(BaseModel):
    NUM_STREAMS: Optional[str] = None
    PERFORMANCE_HINT: Optional[str] = None
    ENABLE_HYPERTHREADING: Optional[bool] = None
    INFERENCE_NUM_THREADS: Optional[str] = None
    PRECISION_HINT: Optional[str] = None

class Payload_Constructor:
    def __init__(self):
        self.generation_config = {}

    def load_model(self, id_model, device, use_cache, export_model, num_streams, performance_hint, precision_hint, is_vision_model, bos_token_id, eos_token_id, pad_token_id, enable_hyperthreading, inference_num_threads, dynamic_shapes): # Added is_vision_model here
        """
        Constructs and sends the load model request based on UI inputs
        
        Args:
            id_model (str): Model identifier or path
            device (str): Device selection for inference
            use_cache (bool): Whether to use cache
            is_vision_model (bool): Whether the model is a vision model
            export_model (bool): Whether to export the model
            num_streams (str): Number of inference streams
            performance_hint (str): Performance optimization strategy
            precision_hint (str): Model precision for computation
            bos_token_id (str): BOS token ID
            eos_token_id (str): EOS token ID
            pad_token_id (str): PAD token ID
            enable_hyperthreading (bool): Whether to enable hyperthreading
            inference_num_threads (str): Number of inference threads
            dynamic_shapes (bool): Whether to use dynamic shapes
        """

        # Create validated load_config
        load_config = LoadModelConfig(
            id_model=id_model,
            use_cache=use_cache,
            device=device,
            export_model=export_model,
            is_vision_model=is_vision_model,   
            eos_token_id=int(eos_token_id) if eos_token_id else None,
            pad_token_id=int(pad_token_id) if pad_token_id else None,
            bos_token_id=int(bos_token_id) if bos_token_id else None,
            dynamic_shapes=dynamic_shapes
        )

        # Create validated ov_config
        ov_config = OVConfig(
            NUM_STREAMS=num_streams if num_streams else None,
            PERFORMANCE_HINT=performance_hint if performance_hint else None,
            ENABLE_HYPERTHREADING=enable_hyperthreading,
            INFERENCE_NUM_THREADS=inference_num_threads if inference_num_threads else None,
            PRECISION_HINT=precision_hint if precision_hint else None
        )

        try:
            response = requests.post(
                f"{OPENARC_URL}/optimum/model/load",
                headers=get_auth_headers(),
                json={
                    "load_config": load_config.model_dump(exclude_none=True),
                    "ov_config": ov_config.model_dump(exclude_none=True)
                }
            )
            response.raise_for_status()
            return response.json(), f"Model loaded successfully: {response.json()}"
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}, f"Error loading model: {str(e)}"

    def unload_model(self):
        """
        Sends an unload model request to the API
        """
        try:
            response = requests.delete(
                f"{OPENARC_URL}/optimum/model/unload",
                headers=get_auth_headers()
            )
            response.raise_for_status()
            return response.json(), f"Model unloaded successfully: {response.json()}"
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}, f"Error unloading model: {str(e)}"

    def status(self):
        """
        Checks the server status
        """
        try:
            response = requests.get(
                f"{OPENARC_URL}/optimum/status",
                headers=get_auth_headers()
            )
            response.raise_for_status()
            return response.json(), f"Server status: {response.json()}"
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}, f"Error checking server status: {str(e)}"
