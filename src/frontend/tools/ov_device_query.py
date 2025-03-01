# Diagnostic Device Query

import openvino as ov
import logging as log
import sys

class DeviceDiagnosticQuery:
    def __init__(self):
        self.core = ov.Core()
        self.available_devices = self.core.available_devices

    def get_available_devices(self):
        """Returns a list of available OpenVINO devices."""
        return self.available_devices

if __name__ == "__main__":
    device_query = DeviceDiagnosticQuery()
    print(device_query.get_available_devices())


    # Device Query: 
# Taken from https://github.com/openvinotoolkit/openvino/blob/master/samples/python/hello_query_device/hello_query_device.py


import openvino as ov

import logging as log
import sys


class DeviceDataQuery:
    def __init__(self):
        self.core = ov.Core()
        
    @staticmethod
    def param_to_string(parameters) -> str:
        """Convert a list / tuple of parameters returned from OV to a string."""
        if isinstance(parameters, (list, tuple)):
            return ', '.join([str(x) for x in parameters])
        return str(parameters)

    def get_available_devices(self) -> list:
        """Return list of available devices."""
        return self.core.available_devices

    def get_device_properties(self, device: str) -> dict:
        """Get all properties for a specific device."""
        properties = {}
        supported_properties = self.core.get_property(device, 'SUPPORTED_PROPERTIES')
        
        for property_key in supported_properties:
            if property_key != 'SUPPORTED_PROPERTIES':
                try:
                    property_val = self.core.get_property(device, property_key)
                    properties[property_key] = self.param_to_string(property_val)
                except TypeError:
                    properties[property_key] = 'UNSUPPORTED TYPE'
        return properties

    def print_device_info(self):
        """Print information about all available devices."""
        log.info('Available devices:')
        for device in self.get_available_devices():
            log.info(f'{device} :')
            log.info('\tSUPPORTED_PROPERTIES:')
            
            properties = self.get_device_properties(device)
            for key, value in properties.items():
                log.info(f'\t\t{key}: {value}')
            log.info('')

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    query = DeviceDataQuery()
    query.print_device_info()
    return 0

if __name__ == '__main__':
    sys.exit(main())