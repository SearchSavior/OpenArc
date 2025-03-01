# Diagnostic Device Query

import openvino as ov

class DiagnosticDeviceQuery:
    def __init__(self):
        self.core = ov.Core()
        self.available_devices = self.core.available_devices

    def get_available_devices(self):
        """Returns a list of available OpenVINO devices."""
        return self.available_devices

if __name__ == "__main__":
    device_query = DiagnosticDeviceQuery()
    print(device_query.get_available_devices())