import gradio as gr

from ..tools.device_query import DeviceDataQuery, DeviceDiagnosticQuery


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
