import gradio as gr
from pathlib import Path

class OpenArc_Documentation:
    """
    The idea of this class is to help keep the documentation organized in a way that will be easy to migrate
    to a new frontend in the future.
    Also, keeping everything in its own md file is probably better for searchability from outside GitHub.
    """
    
    def __init__(self):
        self.doc_components = {}

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