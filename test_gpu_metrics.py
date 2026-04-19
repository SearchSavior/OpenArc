import sys
import json

try:
    import gpu_metrics
except ImportError as e:
    print(f"Error importing gpu_metrics: {e}")
    sys.exit(1)


def test_metrics():
    print("Testing Intel GPU Metrics fetching (Level Zero Sysman)...")
    try:
        data = gpu_metrics.get_gpu_metrics()
        print(json.dumps(data, indent=4))
    except Exception as e:
        print(f"Failed to fetch metrics: {e}")


if __name__ == "__main__":
    test_metrics()
