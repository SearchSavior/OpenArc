import asyncio
import importlib.metadata
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.server.deps import _registry, _workers, verify_api_key
from src.server.downloader import global_downloader
from src.server.models.ov_genai import OVGenAI_GenConfig
from src.server.models.registration import ModelLoadConfig, ModelUnloadConfig
from src.server.models.requests_internal import OpenArcBenchRequest
from src.server.models.requests_management import (
    DownloaderActionRequest,
    DownloaderRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/openarc")


@router.post("/load", dependencies=[Depends(verify_api_key)])
async def load_model(load_config: ModelLoadConfig):
    try:
        model_id = await _registry.register_load(load_config)
        return {
            "model_id": model_id,
            "model_name": load_config.model_name,
            "status": "loaded",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(exc)}")


@router.post("/unload", dependencies=[Depends(verify_api_key)])
async def unload_model(unload_config: ModelUnloadConfig):
    try:
        success = await _registry.register_unload(unload_config.model_name)
        if success:
            return {"model_name": unload_config.model_name, "status": "unloading"}
        else:
            raise HTTPException(
                status_code=404, detail=f"Model '{unload_config.model_name}' not found"
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to unload model: {str(exc)}"
        )


@router.get("/status", dependencies=[Depends(verify_api_key)])
async def get_status():
    return await _registry.status()


class UpdateModelConfigRequest(BaseModel):
    model_path: str
    config: Dict[str, Any]


@router.post("/models/update", dependencies=[Depends(verify_api_key)])
async def update_local_model_config(req: UpdateModelConfigRequest):
    target_path = Path(req.model_path)
    if not target_path.exists() or not target_path.is_dir():
        raise HTTPException(status_code=404, detail="Model directory not found")

    config_path = target_path / "openarc.json"

    current_config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                current_config = json.load(f)
        except Exception:
            pass

    current_config.update(req.config)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(current_config, f, indent=4)
        return {"status": "success", "config": current_config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


@router.get("/models", dependencies=[Depends(verify_api_key)])
async def get_local_models(path: Optional[str] = None):
    if path:
        target_path = Path(path)
    else:
        target_path = Path.home() / ".cache" / "openarc" / "models"

    models = []
    if target_path.exists() and target_path.is_dir():
        for entry in target_path.iterdir():
            if entry.is_dir():
                folder_name = entry.name
                config_path = entry / "openarc.json"
                has_config = config_path.exists()

                model_name = folder_name
                model_type = None
                config_data = {}

                if has_config:
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            config_data = json.load(f)
                            model_name = config_data.get("model_name", model_name)
                            model_type = config_data.get("model_type")
                    except Exception:
                        pass

                models.append(
                    {
                        "id": folder_name,
                        "path": str(entry),
                        "model_name": model_name,
                        "model_type": model_type,
                        "engine": config_data.get("engine"),
                        "vlm_type": config_data.get("vlm_type"),
                        "draft_model_path": config_data.get("draft_model_path"),
                        "draft_device": config_data.get("draft_device"),
                        "num_assistant_tokens": config_data.get("num_assistant_tokens"),
                        "assistant_confidence_threshold": config_data.get(
                            "assistant_confidence_threshold"
                        ),
                        "runtime_config": config_data.get("runtime_config", {}),
                        "has_config": has_config,
                    }
                )
    return {"models": models}


@router.get("/version", dependencies=[Depends(verify_api_key)])
async def get_version():
    try:
        version = importlib.metadata.version("openarc")
    except Exception:
        version = "v0"
    return {"version": version}


def get_hardware_metrics():
    gpus = []
    cpu_info = {"id": "CPU", "name": "System CPU"}
    npus = []

    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        cpu_info["name"] = info.get("brand_raw", "System CPU")
    except Exception as e:
        logging.error(f"Failed to query CPU info: {e}")

    try:
        import openvino as ov

        core = ov.Core()
        devices = core.available_devices
        for device in devices:
            if "NPU" in device:
                try:
                    name = core.get_property(device, "FULL_DEVICE_NAME")
                except Exception:
                    name = device
                npus.append({"id": device, "name": str(name)})
    except Exception:
        pass

    try:
        import gpu_metrics

        data = gpu_metrics.get_gpu_metrics()
        for idx_str, gpu_data in data.items():
            name = gpu_data.get("name", f"Intel GPU {idx_str}")
            total_vram_mb = 0
            used_vram_mb = 0
            is_shared = False

            mem_list = gpu_data.get("memory", [])
            if mem_list and len(mem_list) > 0:
                total_vram_mb = mem_list[0].get("total", 0) // (1024 * 1024)
                used_vram_mb = mem_list[0].get("used", 0) // (1024 * 1024)
            else:
                import psutil

                vm = psutil.virtual_memory()
                total_vram_mb = vm.total // (1024 * 1024)
                is_shared = True

            gpus.append(
                {
                    "id": f"GPU.{idx_str}",
                    "name": str(name),
                    "total_vram": int(total_vram_mb),
                    "used_vram": int(used_vram_mb),
                    "usage": 0.0,
                    "is_shared": is_shared,
                }
            )
    except ImportError:
        logging.warning(
            "gpu_metrics module not found. Intel GPU telemetry will be missing."
        )
    except Exception as e:
        logging.error(f"Failed to fetch GPU metrics: {e}")

    return {"cpu": cpu_info, "gpus": gpus, "npus": npus}


@router.get("/metrics", dependencies=[Depends(verify_api_key)])
async def get_metrics():
    import psutil

    vm = psutil.virtual_memory()
    hw_metrics = await asyncio.to_thread(get_hardware_metrics)

    return {
        "cpus": [
            {
                "id": hw_metrics["cpu"]["id"],
                "name": hw_metrics["cpu"]["name"],
                "cores": psutil.cpu_count(logical=False) or 1,
                "threads": psutil.cpu_count(logical=True) or 1,
                "usage": psutil.cpu_percent(),
            }
        ],
        "total_ram": vm.total // (1024 * 1024),
        "used_ram": vm.used // (1024 * 1024),
        "gpus": hw_metrics["gpus"],
        "npus": hw_metrics["npus"],
    }


@router.post("/downloader", dependencies=[Depends(verify_api_key)])
async def start_download(request: DownloaderRequest):
    try:
        success = await global_downloader.start(request.model_name, request.path)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)},
        )
    if success:
        return {"status": "success", "message": "Model download started successfully."}
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Download already in progress."},
    )


@router.get("/downloader", dependencies=[Depends(verify_api_key)])
async def list_downloads():
    return {"models": global_downloader.list_tasks()}


@router.delete("/downloader", dependencies=[Depends(verify_api_key)])
async def cancel_download(request: DownloaderActionRequest):
    if await global_downloader.cancel(request.model_name):
        return {
            "status": "success",
            "message": "Model download cancelled successfully.",
        }
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Download task not found."},
    )


@router.post("/downloader/pause", dependencies=[Depends(verify_api_key)])
async def pause_download(request: DownloaderActionRequest):
    if await global_downloader.pause(request.model_name):
        return {"status": "success", "message": "Model download paused successfully."}
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Active download task not found."},
    )


@router.post("/downloader/resume", dependencies=[Depends(verify_api_key)])
async def resume_download(request: DownloaderActionRequest):
    if await global_downloader.resume(request.model_name):
        return {"status": "success", "message": "Model download resumed successfully."}
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "No paused download found for this model."},
    )


@router.post("/bench", dependencies=[Depends(verify_api_key)])
async def benchmark(request: OpenArcBenchRequest):
    try:
        config_kwargs = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
            "stream": False,
        }
        if request.input_ids is not None and len(request.input_ids) > 0:
            config_kwargs["input_ids"] = request.input_ids
        else:
            config_kwargs["prompt"] = request.prompt

        config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

        generation_config = OVGenAI_GenConfig(**config_kwargs)

        result = await _workers.generate(request.model, generation_config)
        metrics = result.get("metrics", {}) or {}

        logger.info(
            f"[bench] model={request.model} input_ids_len={len(request.input_ids)} metrics={metrics}"
        )

        return {"metrics": metrics}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(exc)}")
