import argparse
from src.frontend.dashboard import ChatUI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the OpenVINO Chat Dashboard")

    parser.add_argument("--openarc-port", type=int, default=8000,
                        help="Port for the OpenARC server (default: 8000)")
    

    args = parser.parse_args()
    app = ChatUI(openarc_port=args.openarc_port)
    app.launch()

