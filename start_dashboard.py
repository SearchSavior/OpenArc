# OpenArc/start_dashboard.py
import argparse
import gradio as gr
from src.frontend.dashboard import (
    Payload_Constructor, 
    Optimum_Loader, 
    OpenArc_Documentation, 
    DeviceInfoTool,
    ConversionTool,
    update_openarc_url
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the OpenVINO Chat Dashboard")

    parser.add_argument("--openarc-port", type=int, default=8000,
                        help="Port for the OpenARC server (default: 8000)")
    
    args = parser.parse_args()
    # Update OpenARC URL with the provided port
    update_openarc_url(args.openarc_port)
    
    # Create the dashboard components
    payload_constructor = Payload_Constructor()
    
    # Set up the Gradio interface
    with gr.Blocks(title="OpenARC Dashboard") as demo:
        with gr.Tabs():
            # Model loading tab
            optimum_loader = Optimum_Loader(payload_constructor)
            optimum_loader.create_interface()

            # Tools tab with sub-tabs
            with gr.Tab("Tools"):
                with gr.Tabs():
                    with gr.Tab("Model Conversion"):
                        conversion_tool = ConversionTool()
                        conversion_tool.gradio_app()

                    # Device Information tab
                    device_info_tool = DeviceInfoTool()
                    device_info_tool.create_interface()
            
            # Documentation tab
            documentation = OpenArc_Documentation()
            documentation.create_interface()
    
    # Launch the dashboard
    demo.launch()