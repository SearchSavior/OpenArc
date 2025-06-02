import argparse
from src.api.launcher import start_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the OpenVINO Inference API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--openarc-port", type=int, default=8000, 
                        help="Port to bind the server to (default: 8000)")
    args = parser.parse_args()
    start_server(host=args.host, openarc_port=args.openarc_port)