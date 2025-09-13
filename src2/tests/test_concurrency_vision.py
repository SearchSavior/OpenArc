import json
import os
import subprocess
import sys
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"
MAIN_PATH = "/home/echo/Projects/OpenArc/src2/api/main.py"
REQUEST_TIMEOUT_S = int(os.getenv("OPENARC_TEST_REQUEST_TIMEOUT_S", "120"))


@dataclass
class RequestTiming:
    model_name: str
    request_id: int
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_msg: Optional[str] = None
    response_length: int = 0


def http_get(path: str) -> Dict[str, Any]:
    req = Request(f"{BASE_URL}{path}", method="GET")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(f"{BASE_URL}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(timeout_s: int = 21600) -> None:
    start = time.time()
    last_err = None
    while time.time() - start < timeout_s:
        try:
            status = http_get("/openarc/status")
            if isinstance(status, dict):
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"Server did not become ready in {timeout_s}s: {last_err}")


def load_model(model_path: str, model_name: str, device: str) -> str:
    payload = {
        "model_path": model_path,
        "model_name": model_name,
        "model_type": "image_to_text",  # Changed for vision models
        "engine": "ovgenai",
        "device": device,
        "runtime_config": {}
    }
    resp = http_post("/openarc/load", payload)
    return resp.get("model_id", "")


def wait_until_loaded(model_name: str, timeout_s: int = 21600) -> None:
    start = time.time()
    last_err = None
    while time.time() - start < timeout_s:
        try:
            status = http_get("/openarc/status")
            models = status.get("models", [])
            for m in models:
                if m.get("model_name") == model_name and m.get("status") == "loaded":
                    return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(f"Model '{model_name}' did not reach loaded state within {timeout_s}s: {last_err}")


def load_image_as_base64(image_path: str) -> str:
    """Load image file and encode as base64 data URL."""
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"


def generate_once(model_name: str, prompt: str, image_data_url: str, request_id: int, max_new_tokens: int = 64) -> RequestTiming:
    """Generate text from image and return timing information."""
    start_time = time.time()
    timing = RequestTiming(
        model_name=model_name,
        request_id=request_id,
        start_time=start_time,
        end_time=0.0,
        duration=0.0,
        success=False
    )
    
    try:
        payload = {
            "model_name": model_name,
            "gen_config": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_new_tokens": max_new_tokens,
                "stream": True
            }
        }
        response = http_post("/openarc/generate", payload)
        
        timing.end_time = time.time()
        timing.duration = timing.end_time - timing.start_time
        timing.success = True
        timing.response_length = len(response.get("text", ""))
        
        return timing
        
    except Exception as e:
        timing.end_time = time.time()
        timing.duration = timing.end_time - timing.start_time
        timing.success = False
        timing.error_msg = str(e)
        return timing


def print_results(timings: List[RequestTiming], total_elapsed: float) -> None:
    """Print basic test results."""
    successful = [t for t in timings if t.success]
    failed = [t for t in timings if not t.success]
    
    print(f"\nTest Results:")
    print(f"  Total requests: {len(timings)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total time: {total_elapsed:.2f}s")
    
    if successful:
        avg_duration = sum(t.duration for t in successful) / len(successful)
        total_request_time = sum(t.duration for t in successful)
        print(f"  Average request duration: {avg_duration:.2f}s")
        print(f"  Total request time (sequential): {total_request_time:.2f}s")
        print(f"  Speedup: {total_request_time / total_elapsed:.2f}x")
    
    # Show overlapping requests
    overlaps = 0
    for i, t1 in enumerate(timings):
        for t2 in timings[i+1:]:
            if (t1.model_name != t2.model_name and 
                max(t1.start_time, t2.start_time) < min(t1.end_time, t2.end_time)):
                overlaps += 1
    
    print(f"  Overlapping request pairs: {overlaps}")


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    return subprocess.Popen(
        [sys.executable, "-u", MAIN_PATH],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    # Vision model configuration
    model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Qwen/Qwen2.5-VL-7B-Instruct-int4_sym-ov"
    image_path = "/home/echo/Projects/OpenArc/src2/tests/dedication.png"
    model_a = {"name": "Qwen2.5-VL-7B-Instruct-int4_sym-ov-GPU1", "device": "GPU.1"}
    model_b = {"name": "Qwen2.5-VL-7B-Instruct-int4_sym-ov-GPU2", "device": "GPU.2"}

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist: {image_path}")
        return

    # Load image as base64
    print(f"Loading image: {image_path}")
    image_data_url = load_image_as_base64(image_path)
    print("Image loaded and encoded as base64")

    server = start_server()
    try:
        print("Waiting for server to become ready...")
        status_timeout = int(os.getenv("OPENARC_TEST_STATUS_TIMEOUT_S", "21600"))
        wait_for_server(status_timeout)

        print(f"Loading model {model_a['name']} on {model_a['device']}...")
        load_model(model_path, model_a["name"], model_a["device"])
        print(f"Loading model {model_b['name']} on {model_b['device']}...")
        load_model(model_path, model_b["name"], model_b["device"])

        print("Waiting for models to be loaded...")
        load_timeout = int(os.getenv("OPENARC_TEST_LOAD_TIMEOUT_S", "21600"))
        wait_until_loaded(model_a["name"], load_timeout)
        wait_until_loaded(model_b["name"], load_timeout)
        print("Models loaded.")

        # Vision test prompts
        num_requests_per_model = int(os.getenv("OPENARC_TEST_REQUESTS_PER_MODEL", "10"))
        prompt = "Describe what you see in this image in detail."
        max_tokens = 64

        print(f"\nStarting vision concurrency test with {num_requests_per_model} requests per model...")
        
        test_start_time = time.time()
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = []
            
            # Submit requests for both models
            for i in range(num_requests_per_model):
                futures.append(pool.submit(generate_once, model_a["name"], prompt, image_data_url, i, max_tokens))
                futures.append(pool.submit(generate_once, model_b["name"], prompt, image_data_url, i + num_requests_per_model, max_tokens))

            # Collect results
            timings: List[RequestTiming] = []
            for i, fut in enumerate(as_completed(futures)):
                try:
                    timing_result = fut.result()
                    timings.append(timing_result)
                    
                    if timing_result.success:
                        print(f"[{i+1:2d}/{len(futures)}] ✅ {timing_result.model_name}[{timing_result.request_id}]: "
                              f"{timing_result.duration:.2f}s")
                    else:
                        print(f"[{i+1:2d}/{len(futures)}] ❌ {timing_result.model_name}[{timing_result.request_id}]: "
                              f"ERROR: {timing_result.error_msg}")
                        
                except Exception as e:  # noqa: BLE001
                    print(f"[{i+1:2d}/{len(futures)}] ❌ Future error: {e}")

        test_end_time = time.time()
        total_elapsed = test_end_time - test_start_time
        
        print_results(timings, total_elapsed)

    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
