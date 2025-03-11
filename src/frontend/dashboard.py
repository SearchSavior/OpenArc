import gradio as gr
import requests
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import os

from src.frontend.components.model_conversion import ConversionTool
from src.frontend.tools.ov_device_query import DeviceDiagnosticQuery, DeviceDataQuery


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

    def load_model(self, id_model, device, use_cache, export_model, num_streams, performance_hint, precision_hint, bos_token_id, eos_token_id, pad_token_id, enable_hyperthreading, inference_num_threads, dynamic_shapes):
        """
        Constructs and sends the load model request based on UI inputs
        
        Args:
            id_model (str): Model identifier or path
            device (str): Device selection for inference
            use_cache (bool): Whether to use cache
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

class Optimum_Loader:
    def __init__(self, payload_constructor):
        self.payload_constructor = payload_constructor
        self.components = {}

    def read_openvino_config(self, id_model):
        """Read the OpenVINO config file from the model directory and display it in the dashboard as a JSON object.
        This file is generated by the OpenVINO toolkit and contains metadata about the model's configuration and optimizations based on
        how it was converted to the OpenVINO IR.
        """
        try:
            # Convert to Path object and get parent directory
            model_path = Path(id_model)
            config_path = model_path / "openvino_config.json"
            
            if config_path.exists():
                return json.loads(config_path.read_text())
            return {"message": f"No openvino_config.json found in {str(config_path)}"}
        except Exception as e:
            return {"error": f"Error reading config: {str(e)}"}

    def read_architecture(self, id_model):
        """Read the architecture file from the model directory and display it in the dashboard as a JSON object.
        While not explicitly required for inference, this file contains metadata about the model's architecture 
        and can be useful for debugging performance by understanding the model's structure to choose optimization parameters.
        """
        try:
            # Convert to Path object and get parent directory
            model_path = Path(id_model)
            architecture_path = model_path / "config.json"
            
            if architecture_path.exists():
                return json.loads(architecture_path.read_text())
            return {"message": f"No architecture.json found in {str(architecture_path)}"}
        except Exception as e:
            return {"error": f"Error reading architecture: {str(e)}"}
        
    def read_generation_config(self, id_model):
        """Read the generation config file from the model directory and display it in the dashboard as a JSON object.
        This file contains the ground truth of what sampling parameters should be used 
        for inference. These values are dervied directly from the model's pytorch metadata and should be taken as precedent for benchmararks.
        """
        try:
            model_path = Path(id_model)
            generation_config_path = model_path / "generation_config.json"
            
            if generation_config_path.exists():
                return json.loads(generation_config_path.read_text())
            return {"message": f"No generation_config.json found in {str(generation_config_path)}"}
        except Exception as e:
            return {"error": f"Error reading generation config: {str(e)}"}  
        
    def read_tokenizer_config(self, id_model):
        """Read the tokenizer config file from the model directory and display it in the dashboard as a JSON object.
        It might be overkill to include this file in the dashboard, but it's included for completeness.
        OpenArc's goal is to be as flexible as working inside of a script; in that inference scenario you would invedtigate these values and hardcode them.
        Therefore it would not matter if the tokenizer config matches a standardized format for different models. To make OpenArc more scalable, we account for this.
        """
        try:
            model_path = Path(id_model)
            tokenizer_config_path = model_path / "tokenizer_config.json"    
            
            if tokenizer_config_path.exists():
                return json.loads(tokenizer_config_path.read_text())
            return {"message": f"No tokenizer_config.json found in {str(tokenizer_config_path)}"}
        except Exception as e:
            return {"error": f"Error reading tokenizer config: {str(e)}"}   
        
    def create_interface(self):
        with gr.Tab("OpenArc Loader"):
            with gr.Row():
                self.load_model_interface()
                self.debug_tool()
                self.setup_button_handlers()

    def load_model_interface(self):
        with gr.Column(min_width=500, scale=1):
            # Model Basic Configuration
            with gr.Group("Model Configuration"):
                self.components.update({
                    'id_model': gr.Textbox(
                        label="Model Identifier or Path",
                        placeholder="Enter model identifier or local path",
                        info="Enter the model's Hugging Face identifier or local path"
                    ),
                    'device': gr.Dropdown(
                        choices=["", "AUTO", "CPU", "GPU.0", "GPU.1", "GPU.2", "AUTO:GPU.0,GPU.1", "AUTO:GPU.0,GPU.1,GPU.2"],
                        label="Device",
                        value="",
                        info="Select the device for model inference"
                    ),
                    'use_cache': gr.Checkbox(
                        label="Use Cache",
                        value=True,
                        info="Enable cache for stateful models (disable for multi-GPU)"
                    ),
                    'export_model': gr.Checkbox(
                        label="Export Model",
                        value=False,
                        info="Whether to export the model to int8_asym. Default and not recommended."
                    ),
                    'dynamic_shapes': gr.Checkbox(
                        label="Dynamic Shapes",
                        value=True,
                        info="Whether to use dynamic shapes. Default is True. Should only be disabled for NPU inference."
                    )
                })

            # Token Configuration
            with gr.Group("Token Settings"):
                self.components.update({
                    'bos_token_id': gr.Textbox(
                        label="bos_token_id",
                        value="",
                        
                    ),
                    'eos_token_id': gr.Textbox(
                        label="eos_token_id",
                        value="",
                        
                    ),
                    'pad_token_id': gr.Textbox(
                        label="pad_token_id",
                        value="",
                        
                    )
                })

            # Performance Optimization
            with gr.Group("Performance Settings"):
                self.components.update({
                    'num_streams': gr.Textbox(
                        label="Number of Streams",
                        value="",
                        placeholder="Leave empty for default",
                        info="Number of inference streams (optional)"
                    ),
                    'performance_hint': gr.Dropdown(
                        choices=["", "LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"],
                        label="Performance Hint",
                        value="",
                        info="Select performance optimization strategy"
                    ),
                    'precision_hint': gr.Dropdown(
                        choices=["", "auto", "fp16", "fp32", "int8"],
                        label="Precision Hint",
                        value="",
                        info="Select model precision for computation"
                    ),
                    'enable_hyperthreading': gr.Checkbox(
                        label="Enable Hyperthreading",
                        value=True,
                        info="Enable hyperthreading for CPU inference"
                    ),
                    'inference_num_threads': gr.Textbox(
                        label="Inference Number of Threads",
                        value="",
                        placeholder="Leave empty for default",
                        info="Number of inference threads (optional)"
                    )
                })

            # Action Buttons
            with gr.Row():
                self.components.update({
                    'load_button': gr.Button("Load Model", variant="primary"),
                    'unload_button': gr.Button("Unload Model", variant="secondary"),
                    'status_button': gr.Button("Check Status", variant="secondary")
                })

    def debug_tool(self):
        with gr.Column(min_width=300, scale=1):
            with gr.Accordion("Debug Log", open=True):
                self.components['debug_log'] = gr.JSON(
                    label="Log Output",
                    value={"message": "Debug information will appear here..."},
                )
            with gr.Accordion("OpenVINO Config", open=False):
                self.components['config_viewer'] = gr.JSON(
                    label="OpenVINO Configuration",
                    value={"message": "Config will appear here when model path is entered..."},
                )
            with gr.Accordion("Architecture", open=False):
                self.components['architecture_viewer'] = gr.JSON(
                    label="Architecture",
                    value={"message": "Architecture will appear here when model path is entered..."},
                )

            with gr.Accordion("Generation Config", open=False):
                self.components['generation_config_viewer'] = gr.JSON(
                    label="Generation Configuration",
                    value={"message": "Generation config will appear here when model path is entered..."},
                )   

            with gr.Accordion("Tokenizer Config", open=True):
                self.components['tokenizer_config_viewer'] = gr.JSON(
                    label="Tokenizer Configuration",
                    value={"message": "Tokenizer config will appear here when model path is entered..."},
                )   

    def setup_button_handlers(self):
        self.build_load_request()
        
        # Add handler for model path changes
        self.components['id_model'].change(
            fn=self.read_openvino_config,
            inputs=[self.components['id_model']],
            outputs=[self.components['config_viewer']]
        )

        self.components['id_model'].change(
            fn=self.read_architecture,
            inputs=[self.components['id_model']],
            outputs=[self.components['architecture_viewer']]
        )

        self.components['id_model'].change(
            fn=self.read_generation_config,
            inputs=[self.components['id_model']],
            outputs=[self.components['generation_config_viewer']]
        )

        self.components['id_model'].change(
            fn=self.read_tokenizer_config,
            inputs=[self.components['id_model']],
            outputs=[self.components['tokenizer_config_viewer']]
        )
        
    def build_load_request(self):
        self.components['load_button'].click(
            fn=self.payload_constructor.load_model,
            inputs=[
                self.components[key] for key in [
                    'id_model', 'device', 'use_cache', 'export_model',
                    'num_streams', 'performance_hint', 'precision_hint',
                    'bos_token_id', 'eos_token_id', 'pad_token_id',
                    'enable_hyperthreading', 'inference_num_threads', 'dynamic_shapes'
                ]
            ],
            outputs=[self.components['debug_log']]
        )
        
        self.components['unload_button'].click(
            fn=self.payload_constructor.unload_model,
            inputs=None,
            outputs=[self.components['debug_log']]
        )

        self.components['status_button'].click(
            fn=self.payload_constructor.status,
            inputs=None,
            outputs=[self.components['debug_log']]
        )

class OpenArc_Documentation:
    def __init__(self):
        self.doc_components = {}
        # Organize documentation files into categories
        self.doc_categories = {
            "Performance Hints": [
                "LATENCY",
                "THROUGHPUT",
                "CUMULATIVE_THROUGHPUT"
            ],
            "CPU Options": [
                "Enable Hyperthreading",
                "Inference Num Threads",
                "Scheduling Core Type"
            ],
            "Streaming Options": [
                "Num Streams"
            ]
        }
        
        # Map topic names to file paths
        self.doc_files = {
            "LATENCY": "docs/ov_config/performance_hint_latency.md",
            "THROUGHPUT": "docs/ov_config/performance_hint_throughput.md",
            "CUMULATIVE_THROUGHPUT": "docs/ov_config/performance_hint_cumulative_throughput.md",
            "Enable Hyperthreading": "docs/ov_config/enable_hyperthreading.md",
            "Inference Num Threads": "docs/ov_config/inference_num_threads.md",
            "Num Threads": "docs/ov_config/num_threads.md",
            "Num Streams": "docs/ov_config/num_streams.md",
            "Scheduling Core Type": "docs/ov_config/scheduling_core_type.md"
        }
        
    def read_markdown_file(self, file_path):
        """Read a markdown file and return its contents"""
        try:
            path = Path(file_path)
            if path.exists():
                return path.read_text()
            return f"Documentation file not found: {file_path}"
        except Exception as e:
            return f"Error reading documentation: {str(e)}"
    
    def display_doc(self, doc_name):
        """Display the selected documentation"""
        if doc_name in self.doc_files:
            return self.read_markdown_file(self.doc_files[doc_name])
        return "Please select a documentation topic from the list."
    
    def create_interface(self):
        with gr.Tab("Documentation"):
            with gr.Row():
                gr.Markdown("# OpenArc Documentation")
            
            with gr.Row():
                # Create columns for layout
                nav_col = gr.Column(scale=1)
                content_col = gr.Column(scale=3)
                
                # Create the content markdown component first
                with content_col:
                    doc_content = gr.Markdown(
                        value="""
# OpenVINO Configuration Documentation

Welcome to the OpenArc documentation for OpenVINO configuration options. 
This documentation will help you understand how to optimize your model inference using various configuration parameters.

## Getting Started

Select a topic from the navigation panel on the left to view detailed documentation.

The configuration options are organized into categories:
- **Performance Hints**: Options that control the performance optimization strategy
- **CPU Options**: Settings specific to CPU execution
- **Streaming Options**: Parameters for controlling inference streams
- **Scheduling Options**: Options for thread scheduling and core allocation
"""
                    )
                    # Store the component for later reference
                    self.doc_components['doc_content'] = doc_content
                
                # Now create the navigation sidebar with buttons
                with nav_col:
                    gr.Markdown("## Configuration Options")
                    
                    # Create accordions for each category
                    for category, topics in self.doc_categories.items():
                        with gr.Accordion(f"{category} ({len(topics)})", open=True):
                            for topic in topics:
                                topic_btn = gr.Button(topic, size="sm")
                                # Set up click handler for each button
                                topic_btn.click(
                                    fn=self.display_doc,
                                    inputs=[gr.Textbox(value=topic, visible=False)],
                                    outputs=[self.doc_components['doc_content']]
                                )

class DeviceInfoTool:
    def __init__(self):
        self.device_data_query = DeviceDataQuery()
        self.device_diagnostic_query = DeviceDiagnosticQuery()
        
    def get_available_devices(self):
        """Get list of available devices from DeviceDiagnosticQuery"""
        devices = self.device_diagnostic_query.get_available_devices()
        return {"Available Devices": devices}
    
    def get_device_properties(self):
        """Get detailed properties for all available devices from DeviceDataQuery"""
        devices = self.device_data_query.get_available_devices()
        result = {}
        
        for device in devices:
            properties = self.device_data_query.get_device_properties(device)
            result[device] = properties
            
        return result
    
    def create_interface(self):
        with gr.Tab("Devices"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## Available Devices")
                    device_list = gr.JSON(label="Device List")
                    refresh_button = gr.Button("Refresh Device List")
                    refresh_button.click(
                        fn=self.get_available_devices,
                        inputs=[],
                        outputs=[device_list]
                    )
                with gr.Column(scale=2):
                    gr.Markdown("## Device Properties")
                    device_properties = gr.JSON(label="Device Properties")
                    properties_button = gr.Button("Get Device Properties")
                    properties_button.click(
                        fn=self.get_device_properties,
                        inputs=[],
                        outputs=[device_properties]
                    )



