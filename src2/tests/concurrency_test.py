import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"
MAIN_PATH = "/home/echo/Projects/OpenArc/src2/api/main.py"
REQUEST_TIMEOUT_S = int(os.getenv("OPENARC_TEST_REQUEST_TIMEOUT_S", "120"))


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
        "model_type": "text_to_text",
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


def generate_once(model_name: str, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7) -> Dict[str, Any]:
    payload = {
        "model_name": model_name,
        "gen_config": {
            "messages": [{"role": "user", "content": prompt}],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False
        }
    }
    return http_post("/openarc/generate", payload)


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    # Unbuffered output for immediate logs
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
    # Update these paths for your environment if needed
    model_path = "/mnt/Ironwolf-4TB/Models/OpenVINO/Mistral/Impish_Nemo_12B-int4_asym-awq-ov"
    model_a = {"name": "Impish_Nemo_GPU1", "device": "GPU.1"}
    model_b = {"name": "Impish_Nemo_GPU2", "device": "GPU.2"}

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

        prompts_a = [f"Hello from A #{i+1}" for i in range(5)]
        prompts_b = [f"Hello from B #{i+1}" for i in range(5)]

        futures = []
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as pool:
            for p in prompts_a:
                futures.append(pool.submit(generate_once, model_a["name"], p))
            for p in prompts_b:
                futures.append(pool.submit(generate_once, model_b["name"], p))

            results: List[Dict[str, Any]] = []
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    results.append(res)
                    text = res.get("text", "")
                    print(f"[OK] len={len(text)}")
                except Exception as e:  # noqa: BLE001
                    print(f"[ERR] {e}")

        elapsed = time.time() - start_time
        print(f"Completed {len(futures)} requests in {elapsed:.2f}s")

    finally:
        stop_server(server)


if __name__ == "__main__":
    main()


