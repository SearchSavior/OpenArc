#!/usr/bin/env python3
"""
OpenArc CLI Tool - Command-line interface for OpenArc model loading operations.
"""
import traceback
import argparse
import json
import os
import sys
import requests

# Import device query classes
try:
    from src.cli.device_query import DeviceDataQuery, DeviceDiagnosticQuery
except ImportError:
    DeviceDataQuery = None
    DeviceDiagnosticQuery = None


class OpenArcCLI:
    def __init__(self):
        self.base_url = os.getenv('OPENARC_API_URL', 'http://localhost:8000')
        self.api_key = os.getenv('OPENARC_API_KEY', '')
        
    def get_headers(self):
        """Get headers for API requests."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    def load_model(self, args):
        """Load a model using the /optimum/model/load endpoint."""
        # Build load_config from arguments
        load_config = {
            "id_model": args.model,
            "model_type": args.model_type,
            "use_cache": args.use_cache,
            "device": args.device,
            "dynamic_shapes": args.dynamic_shapes,
            "export_model": args.export_model,
        }
        
        # Add optional token IDs if provided
        if args.pad_token_id is not None:
            load_config["pad_token_id"] = args.pad_token_id
        if args.eos_token_id is not None:
            load_config["eos_token_id"] = args.eos_token_id
        if args.bos_token_id is not None:
            load_config["bos_token_id"] = args.bos_token_id
        
        # Build ov_config from arguments
        ov_config = {}
        if args.PERFORMANCE_HINT:
            ov_config["PERFORMANCE_HINT"] = args.PERFORMANCE_HINT
        if args.INFERENCE_PRECISION_HINT:
            ov_config["INFERENCE_PRECISION_HINT"] = args.INFERENCE_PRECISION_HINT
        if args.ENABLE_HYPER_THREADING is not None:
            ov_config["ENABLE_HYPER_THREADING"] = args.ENABLE_HYPER_THREADING
        if args.INFERENCE_NUM_THREADS is not None:
            ov_config["INFERENCE_NUM_THREADS"] = args.INFERENCE_NUM_THREADS
        if args.SCHEDULING_CORE_TYPE:
            ov_config["SCHEDULING_CORE_TYPE"] = args.SCHEDULING_CORE_TYPE
        if args.NUM_STREAMS:
            ov_config["NUM_STREAMS"] = args.NUM_STREAMS
        
        # Prepare payload
        payload = {
            "load_config": load_config,
            "ov_config": ov_config if ov_config else {}
        }
        
        if args.dry_run:
            print("🔍 Dry run - would send the following payload:")
            print(json.dumps(payload, indent=2))
            return 0
        
        # Make API request
        url = f"{self.base_url}/optimum/model/load"
        
        try:
            print(f"🚀 Loading model: {args.model}")
            response = requests.post(url, json=payload, headers=self.get_headers())
            
            if response.status_code == 200:
                print("✅ Model loaded successfully!")
                if args.verbose:
                    print("Response:", json.dumps(response.json(), indent=2))
            else:
                print(f"❌ Error loading model: {response.status_code}")
                print("Response:", response.text)
                return 1
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return 1
        
        return 0

    def unload_model(self, args):
        """Unload a model using the /optimum/model/unload endpoint."""
        if args.dry_run:
            print(f"🔍 Dry run - would unload model: {args.model_id}")
            return 0
        
        # Make API request
        url = f"{self.base_url}/optimum/model/unload"
        params = {"model_id": args.model_id}
        
        try:
            print(f"🗑️  Unloading model: {args.model_id}")
            response = requests.delete(url, params=params, headers=self.get_headers())
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {result['message']}")
                if args.verbose:
                    print("Response:", json.dumps(result, indent=2))
            else:
                print(f"❌ Error unloading model: {response.status_code}")
                print("Response:", response.text)
                return 1
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return 1
        
        return 0

    def status(self, args):
        """Get status of loaded models using the /optimum/status endpoint."""
        if args.dry_run:
            print("🔍 Dry run - would get status")
            return 0
        
        # Make API request
        url = f"{self.base_url}/optimum/status"
        
        try:
            print("📊 Getting model status...")
            response = requests.get(url, headers=self.get_headers())
            
            if response.status_code == 200:
                result = response.json()
                loaded_models = result.get("loaded_models", {})
                total_models = result.get("total_models_loaded", 0)
                
                print(f"\n📈 Status Report - {total_models} model(s) loaded:")
                print("-" * 60)
                
                if not loaded_models:
                    print("No models currently loaded.")
                else:
                    for model_id, model_info in loaded_models.items():
                        device = model_info.get("device", "unknown")
                        status = model_info.get("status", "unknown")
                        metadata = model_info.get("model_metadata", {})
                        model_type = metadata.get("model_type", "unknown")
                        perf_hint = metadata.get("PERFORMANCE_HINT", "none")
                        
                        print(f"🔹 Model ID: {model_id}")
                        print(f"   Status: {status}")
                        print(f"   Device: {device}")
                        print(f"   Type: {model_type}")
                        print(f"   Performance Hint: {perf_hint}")
                        print()
                
                if args.verbose:
                    print("Full Response:")
                    print(json.dumps(result, indent=2))
                    
            else:
                print(f"❌ Error getting status: {response.status_code}")
                print("Response:", response.text)
                return 1
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return 1
        
        return 0

    def device_data(self, args):
        """Query device properties and information for all devices."""
        if DeviceDataQuery is None:
            print("❌ Device query functionality not available. Make sure src.cli.device_query is accessible.")
            return 1
        
        if args.dry_run:
            print("🔍 Dry run - would query device data for all devices")
            return 0
        
        try:
            print("🔍 Querying device data for all devices...")
            device_query = DeviceDataQuery()
            available_devices = device_query.get_available_devices()
            
            print(f"\n📊 Available Devices ({len(available_devices)}):")
            print("-" * 60)
            
            if not available_devices:
                print("❌ No devices found!")
                return 1
            
            for device in available_devices:
                print(f"\n🔹 Device: {device}")
                print("   SUPPORTED_PROPERTIES:")
                
                properties = device_query.get_device_properties(device)
                for key, value in properties.items():
                    print(f"      {key}: {value}")
                
            print(f"\n✅ Found {len(available_devices)} device(s)")
            
            if args.verbose:
                print("\nDetailed device list:")
                print(json.dumps(available_devices, indent=2))
                
            return 0
            
        except Exception as e:
            print(f"❌ Error querying device data: {e}")
            if args.verbose:
                traceback.print_exc()
            return 1

    def device_diagnose(self, args):
        """Diagnose available OpenVINO devices."""
        if DeviceDiagnosticQuery is None:
            print("❌ Device diagnostic functionality not available. Make sure src.cli.device_query is accessible.")
            return 1
        
        if args.dry_run:
            print("🔍 Dry run - would diagnose devices")
            return 0
        
        try:
            print("🔍 Diagnosing OpenVINO devices...")
            diagnostic = DeviceDiagnosticQuery()
            available_devices = diagnostic.get_available_devices()
            
            print("-" * 60)
            print(f"📋 Available Devices: {len(available_devices)}")
            
            if not available_devices:
                print("❌ No OpenVINO devices found!")
                return 1
            
            for i, device in enumerate(available_devices, 1):
                print(f"🔹 Device {i}: {device}")
            
            print(f"\n✅ OpenVINO runtime found {len(available_devices)} device(s)")
                
            return 0
            
        except Exception as e:
            print(f"❌ Error during device diagnosis: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description='OpenArc CLI - Load and manage models with OpenVINO optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Thank's for checking out this project. It means a lot. 
---

Welcome to the OpenArc CLI!

This tool makes it easier to inferface with the OpenArc server. 


- Load models into the OpenArc server
- Check the status of loaded models
- Unload models
- Query device properties
- Query installed devices


Examples:
  # Load a text model onto GPU.0, which is always the first GPU
  %(prog)s python openarc-cli.py load \
    --model "path/to/text/model"   \
    --model_type TEXT   \
    --use_cache  \
    --device GPU.0   \
    --dynamic_shapes   \
    --pad_token_id int \
    --bos_token_id int \
    --eos_token_id int \
    --PERFORMANCE_HINT LATENCY

  # Load a vision model onto GPU.0  
   %(prog)s python openarc-cli.py load \
    --model "path/to/vision/model"   \
    --model_type VISION   \
    --use_cache  \
    --device GPU.0   \
    --dynamic_shapes   \
    --pad_token_id int \
    --bos_token_id int \
    --eos_token_id int \
    --PERFORMANCE_HINT LATENCY


  # Get status of loaded models
    - Sends GET to /v1/models
    - Returns a list of model-ids (id_model in the src)
    - 
  %(prog)s python openarc-cli.py status

  # Unload a specific model
  %(prog)s python openarc-cli.py unload --model-id "model-name" [this should be one of the model-ids returned by the status command]

  # Query all device properties. 
    - A report detailing supported performance properties
    - These are device specific and cannot be changed 
    - The runtime uses these to determine how to use OpenVINO optimizations... optimally.
    - IN this sense, most of the performance properties are actually *over-writing* the default values seen here 

  %(prog)s python openarc-cli.py device-data

  # Diagnose available devices
    - Use to check if your devices are detected. Useful for debugging driver issues.
  %(prog)s python openarc-cli.py device-diagnose

Environment Variables:
  OPENARC_API_URL    API base URL (default: http://localhost:8000)
  OPENARC_API_KEY    API authentication key"""
    )
    
    # Global options
    parser.add_argument('--api-url', default=os.getenv('OPENARC_API_URL', 'http://localhost:8000'),
                       help='API base URL')
    parser.add_argument('--api-key', default=os.getenv('OPENARC_API_KEY', ''),
                       help='API authentication key')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be sent without making API calls')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load a model')
    load_parser.add_argument('--model', required=True, help='Model identifier or path')
    load_parser.add_argument('--model_type', choices=['TEXT', 'VISION'], default='TEXT',
                           help='Type of model (default: TEXT)')
    load_parser.add_argument('--device', default='CPU',
                           help='Device: CPU, GPU.0, AUTO (default: CPU)')
    load_parser.add_argument('--use_cache', action='store_true', default=True,
                           help='Use cache for stateful models (default: True)')
    load_parser.add_argument('--no-use_cache', dest='use_cache', action='store_false',
                           help='Disable cache for stateful models')
    load_parser.add_argument('--export_model', action='store_true', default=False,
                           help='Export the model')
    load_parser.add_argument('--dynamic_shapes', action='store_true', default=True,
                           help='Use dynamic shapes (default: True)')
    load_parser.add_argument('--no-dynamic_shapes', dest='dynamic_shapes', action='store_false',
                           help='Use static shapes')
    
    # Token configuration for load command
    load_parser.add_argument('--pad_token_id', type=int, help='Custom pad token ID')
    load_parser.add_argument('--eos_token_id', type=int, help='Custom end of sequence token ID')
    load_parser.add_argument('--bos_token_id', type=int, help='Custom beginning of sequence token ID')
    
    # OpenVINO optimization options for load command
    load_parser.add_argument('--NUM_STREAMS', default=None, help='Number of inference streams')
    load_parser.add_argument('--PERFORMANCE_HINT', default=None, choices=['LATENCY', 'THROUGHPUT', 'CUMULATIVE_THROUGHPUT'],
                           help='Performance optimization hint')
    load_parser.add_argument('--INFERENCE_PRECISION_HINT', default=None, choices=['auto', 'fp32', 'fp16', 'int8'],
                           help='Inference precision')
    load_parser.add_argument('--ENABLE_HYPER_THREADING', default=None, type=lambda x: x.lower() == 'true',
                           help='Enable hyper-threading (true/false)')
    load_parser.add_argument('--INFERENCE_NUM_THREADS', default=None, type=int,
                           help='Number of inference threads')
    load_parser.add_argument('--SCHEDULING_CORE_TYPE', default=None, choices=['ANY_CORE', 'PCORE_ONLY', 'ECORE_ONLY'],
                           help='Core scheduling type')
    
    # Unload command
    unload_parser = subparsers.add_parser('unload', help='Unload a model')
    unload_parser.add_argument('--model-id', dest='model_id', required=True, 
                             help='Model ID to unload (use status command to see loaded models)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get status of loaded models')
    status_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    status_parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')

    # Device data subcommand
    device_data_parser = subparsers.add_parser('device-data', help='Query OpenVINO device properties for all devices')
    device_data_parser.set_defaults(command='device_data')
    device_data_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    device_data_parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')

    # Device diagnose subcommand
    device_diagnose_parser = subparsers.add_parser('device-diagnose', help='Diagnose available OpenVINO devices')
    device_diagnose_parser.set_defaults(command='device_diagnose')
    device_diagnose_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    device_diagnose_parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')

    return parser

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create CLI instance
    cli = OpenArcCLI()
    cli.base_url = args.api_url
    cli.api_key = args.api_key
    
    # Execute the appropriate command
    try:
        if args.command == 'load':
            return cli.load_model(args)
        elif args.command == 'unload':
            return cli.unload_model(args)
        elif args.command == 'status':
            return cli.status(args)
        elif args.command == 'device_data':
            return cli.device_data(args)
        elif args.command == 'device_diagnose':
            return cli.device_diagnose(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())



