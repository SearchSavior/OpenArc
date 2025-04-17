import gradio as gr
import pandas as pd


class ModelManager:
    def __init__(self, payload_constructor):
        self.payload_constructor = payload_constructor
        self.components = {}

    def _refresh_models(self):
        """Helper function to fetch and format loaded models data"""
        response, _ = self.payload_constructor.status()
        
        if "error" in response:
            return pd.DataFrame(), "Error fetching model status"
        
        loaded_models = response.get("loaded_models", {})
        total_models = response.get("total_models_loaded", 0)
        
        # Format data for two-column DataFrame
        model_data = []
        for model_name, model_info in loaded_models.items():
            metadata = model_info.get("model_metadata", {})
            
            # Add model header row
            model_data.append({"Attribute": f"{model_name}", "Value": ""})
            
            # Add all model attributes
            model_data.append({"Attribute": "Status", "Value": model_info.get("status", "")})
            model_data.append({"Attribute": "Device", "Value": model_info.get("device", "")})
            model_data.append({"Attribute": "Path", "Value": metadata.get("id_model", "")})
            model_data.append({"Attribute": "use_cache", "Value": metadata.get("use_cache", "")})
            model_data.append({"Attribute": "dynamic_shapes", "Value": metadata.get("dynamic_shapes", "")})
            model_data.append({"Attribute": "pad_token_id", "Value": metadata.get("pad_token_id", "")})
            model_data.append({"Attribute": "eos_token_id", "Value": metadata.get("eos_token_id", "")})
            model_data.append({"Attribute": "bos_token_id", "Value": metadata.get("bos_token_id", "")})
            model_data.append({"Attribute": "is_vision_model", "Value": metadata.get("is_vision_model", "")})
            model_data.append({"Attribute": "is_text_model", "Value": metadata.get("is_text_model", "")})
            model_data.append({"Attribute": "NUM_STREAMS", "Value": metadata.get("NUM_STREAMS", "")})
            model_data.append({"Attribute": "PERFORMANCE_HINT", "Value": metadata.get("PERFORMANCE_HINT", "")})
            model_data.append({"Attribute": "PRECISION_HINT", "Value": metadata.get("PRECISION_HINT", "")})
            model_data.append({"Attribute": "ENABLE_HYPER_THREADING", "Value": metadata.get("ENABLE_HYPER_THREADING", "")})
            model_data.append({"Attribute": "INFERENCE_NUM_THREADS", "Value": metadata.get("INFERENCE_NUM_THREADS", "")})
            model_data.append({"Attribute": "SCHEDULING_CORE_TYPE", "Value": metadata.get("SCHEDULING_CORE_TYPE", "")})
            
            # Add empty row between models
            model_data.append({"Attribute": "", "Value": ""})
        
        df = pd.DataFrame(model_data)
        status_text = f"Total Models Loaded: {total_models}"
        return df, status_text

    def _unload_model(self):
        """Helper function to unload a model"""
        response, _ = self.payload_constructor.unload_model()
        return response

    def _unload_model_ui(self, model_id):
        """Helper function to handle model unloading"""
        _, status_msg = self.payload_constructor.unload_model(model_id)
        return status_msg

    def create_interface(self):
        with gr.Tab("Model Manager"):
            gr.Markdown("## Model Management Interface")
            
            with gr.Row():
                refresh_btn = gr.Button("Refresh Loaded Models")
                status_text = gr.Textbox(label="Status", interactive=False)
            
            model_table = gr.DataFrame(
                headers=["Attribute", "Value"],
                datatype=["str", "str"],
                interactive=False,
                wrap=True,
            )
            
            refresh_btn.click(
                fn=self._refresh_models,
                outputs=[model_table, status_text]
            )

            with gr.Row():
                model_id_input = gr.Textbox(label="Model ID to Unload")
                unload_btn = gr.Button("Unload Model")
                unload_status = gr.Textbox(label="Unload Status", interactive=False)
            
            unload_btn.click(
                fn=self._unload_model_ui,
                inputs=model_id_input,
                outputs=unload_status
            )
