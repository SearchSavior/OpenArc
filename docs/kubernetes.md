---
icon: lucide/ship-wheel
---

OpenARC can be deployed to Kubernetes. Here are some examples

=== "Simple deployment"

    This is a relatively simple deployment for running Qwen3-1.7B on a CPU. The model, device, and other parameters can be changed by modifying the initcontainer environment variables.

    ```yaml
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: &app_name openarc
      labels: &labels
        app: *app_name
    spec:
      selector:
        matchLabels: *labels
      template:
        metadata:
          labels: *labels
        spec:
          initContainers:
            - name: prepare-model
              image: &image ghcr.io/searchsavior/openarc
              env:
                - name: MODEL_NAME
                  value: &model_name qwen3
                - name: MODEL_TYPE
                  value: llm
                - name: ENGINE_NAME
                  value: ovgenai
                - name: DEVICE
                  value: CPU
                - name: HF_REPO_NAME
                  value: Echo9Zulu/Qwen3-1.7B-int8_asym-ov
                - &config_file_var
                  name: OPENARC_CONFIG_FILE
                  value: /models/openarc_config.json
                # Optional extra args, see https://searchsavior.github.io/OpenArc/commands/#add
                # - name: EXTRA_ADD_ARGS
                #   value: "--runtime-config '{\"MODEL_DISTRIBUTION_POLICY\": \"PIPELINE_PARALLEL\"}'"
              command: ["python", "-c"]
              args:
                - |
                  import shlex, os
                  from huggingface_hub import snapshot_download

                  model_name = os.environ["MODEL_NAME"]

                  # Download the model if it doesn't exist already
                  model_dir = f"/models/{model_name}"
                  if not os.path.exists(f"{model_dir}/openvino_model.xml"):
                      snapshot_download(repo_id=os.environ["HF_REPO_NAME"], local_dir=model_dir)

                  # Register the model with OpenArc. The runtime will autoload it on startup.
                  add_command = [
                      "openarc", "add",
                      # Must match the OPENARC_AUTOLOAD_MODEL env var in the openarc container.
                      "--model-name", model_name,
                      "--model-path", model_dir,
                      "--engine", os.environ["ENGINE_NAME"],
                      "--model-type", os.environ["MODEL_TYPE"],
                      "--device", os.environ["DEVICE"]
                  ]
                  if "EXTRA_ADD_ARGS" in os.environ:
                      add_command.extend(shlex.split(os.environ["EXTRA_ADD_ARGS"]))

                  os.execlp(add_command[0], *add_command)
              volumeMounts: &mounts
                - name: models
                  mountPath: /models
          containers:
            - name: openarc
              image: *image
              env:
                - name: OPENARC_AUTOLOAD_MODEL
                  value: *model_name
                - *config_file_var
              ports:
                - containerPort: 8000
                  name: http
              # TODO add probes once I fix the /status endpoint bug
              volumeMounts: *mounts
          volumes:
            # Holds the model itself
            - name: models
              emptyDir: {}
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: &app_name openarc
      labels: &labels
        app: *app_name
    spec:
      selector: *labels
      ports:
        - name: http
          port: 8000
    ```

=== "GPU access via the Intel GPU DRA plugin for Kubernetes"

    OpenARC works with the [Intel GPU DRA plugin for Kubernetes](https://github.com/intel/intel-resource-drivers-for-kubernetes). This allows pods to only be scheduled on nodes
    with Intel GPUs available, and allows choosing which GPUs to make available to OpenARC when nodes contain multiple GPUs. This service is not required for GPU access - you can
    mount a hostpath to the device, or use the older [Intel GPU device plugin for Kubernetes (non-DRA)](https://github.com/intel/intel-device-plugins-for-kubernetes/blob/main/cmd/gpu_plugin/README.md).

    This requires Kubernetes 1.34 or newer.

    ```yaml
    ---
    apiVersion: resource.k8s.io/v1
    kind: ResourceClaimTemplate
    metadata:
      name: openarc-gpu
      labels:
        app: openarc
    spec:
      spec:
        devices:
          requests:
            - name: gpu
              exactly:
                deviceClassName: gpu.intel.com
                selectors:
                  - cel:
                      # Configure this for your device
                      expression: device.attributes["gpu.intel.com"].family == 'Arc Pro'
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: &app_name openarc
      labels: &labels
        app: *app_name
    spec:
      selector:
        matchLabels: *labels
      template:
        metadata:
          labels: *labels
        spec:
          initContainers:
            - name: prepare-model
              image: &image ghcr.io/searchsavior/openarc
              env:
                - name: MODEL_NAME
                  value: &model_name qwen3
                - name: MODEL_TYPE
                  value: llm
                - name: ENGINE_NAME
                  value: ovgenai
                - name: DEVICE
                  value: GPU
                - name: HF_REPO_NAME
                  value: Echo9Zulu/Qwen3-1.7B-int8_asym-ov
                - &config_file_var
                  name: OPENARC_CONFIG_FILE
                  value: /models/openarc_config.json
                # Optional extra args, see https://searchsavior.github.io/OpenArc/commands/#add
                # - name: EXTRA_ADD_ARGS
                #   value: "--runtime-config '{\"MODEL_DISTRIBUTION_POLICY\": \"PIPELINE_PARALLEL\"}'"
              command: ["python", "-c"]
              args:
                - |
                  import shlex, os
                  from huggingface_hub import snapshot_download

                  model_name = os.environ["MODEL_NAME"]

                  # Download the model if it doesn't exist already
                  model_dir = f"/models/{model_name}"
                  if not os.path.exists(f"{model_dir}/openvino_model.xml"):
                      snapshot_download(repo_id=os.environ["HF_REPO_NAME"], local_dir=model_dir)

                  # Register the model with OpenArc. The runtime will autoload it on startup.
                  add_command = [
                      "openarc", "add",
                      # Must match the OPENARC_AUTOLOAD_MODEL env var in the openarc container.
                      "--model-name", model_name,
                      "--model-path", model_dir,
                      "--engine", os.environ["ENGINE_NAME"],
                      "--model-type", os.environ["MODEL_TYPE"],
                      "--device", os.environ["DEVICE"]
                  ]
                  if "EXTRA_ADD_ARGS" in os.environ:
                      add_command.extend(shlex.split(os.environ["EXTRA_ADD_ARGS"]))

                  os.execlp(add_command[0], *add_command)
              volumeMounts: &mounts
                - name: models
                  mountPath: /models
          containers:
            - name: openarc
              image: *image
              env:
                - name: OPENARC_AUTOLOAD_MODEL
                  value: *model_name
                - *config_file_var
              ports:
                - containerPort: 8000
                  name: http
              # TODO add probes once I fix the /status endpoint bug
              volumeMounts: *mounts
              # Important: If building a model caching via a separate pod, the GPU must also be made available to the caching pod.
              resources:
                claims:
                  - name: gpu
          volumes:
            # Holds the model itself
            - name: models
              emptyDir: {}
          resourceClaims:
            - name: gpu
              resourceClaimTemplateName: openarc-gpu
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: &app_name openarc
      labels: &labels
        app: *app_name
    spec:
      selector: *labels
      ports:
        - name: http
          port: 8000
    ```

=== "Loading images via imageVolumes"

    Kubernetes 1.33 and above supports mounting OCI container images as volumes. This can be used to avoid download/setup initcontainers and jobs. Users are expected to build and publish
    their own container images for this purpose.

    Note that it is not generally recommended to cache compiled models within model container images. Compiled models are dependent on device, kernel, kernel module, OpenArc, and model
    versions. Changing any one of these invalidates the cache, resulting in models not being portable. If you choose to cache compiled models in a container image, make sure that they are
    built against the exact dependency versions (including GPU) that will be used within the Kubernetes cluster.

    Example Dockerfile for building a Qwen3-1.7B image:

    ```Dockerfile
    # syntax=docker/dockerfile:1.7
    # Build a self-contained model image: just the downloaded HF model directory
    # plus a generated openarc_config.json with a *relative* model_path so the
    # image's contents can be mounted into the openarc runtime at any location.

    ARG OPENARC_IMAGE=ghcr.io/searchsavior/openarc

    FROM ${OPENARC_IMAGE} AS builder

    ARG MODEL_NAME=qwen3
    ARG MODEL_TYPE=llm
    ARG ENGINE_NAME=ovgenai
    ARG DEVICE=CPU
    ARG HF_REPO_NAME=Echo9Zulu/Qwen3-1.7B-int8_asym-ov

    ENV MODEL_NAME=${MODEL_NAME} \
        MODEL_TYPE=${MODEL_TYPE} \
        ENGINE_NAME=${ENGINE_NAME} \
        DEVICE=${DEVICE} \
        HF_REPO_NAME=${HF_REPO_NAME}

    WORKDIR /out

    RUN <<'PY' python
    import os, shutil, subprocess
    from huggingface_hub import snapshot_download

    model_name = os.environ["MODEL_NAME"]
    model_dir = f"/out/{model_name}"

    # Download the model into /out/<model_name>, then drop HF's bookkeeping cache.
    snapshot_download(repo_id=os.environ["HF_REPO_NAME"], local_dir=model_dir)
    shutil.rmtree(f"{model_dir}/.cache", ignore_errors=True)

    # Generate the config alongside the model so model_path can be a relative
    # basename — openarc resolves it against the config file's directory both
    # at write-time validation and at runtime, making the image location-independent.
    subprocess.run(
        [
            "openarc", "add",
            "--model-name", model_name,
            "--model-path", model_name,
            "--engine", os.environ["ENGINE_NAME"],
            "--model-type", os.environ["MODEL_TYPE"],
            "--device", os.environ["DEVICE"],
        ],
        check=True,
        env={**os.environ, "OPENARC_CONFIG_FILE": "/out/openarc_config.json"},
    )
    PY

    # Final image: only the model directory and config file, nothing else.
    FROM scratch
    COPY --from=builder /out/ /
    ```

    Build and push the image:

    ```console
    $ docker build -f openarc-model.Dockerfile -t ghcr.io/<you>/openarc-qwen3-model:latest .
    $ docker push ghcr.io/<you>/openarc-qwen3-model:latest
    ```

    Or to package a different model from HuggingFace, override the build args:

    ```console
    $ docker build \
        -f openarc-model.Dockerfile \
        --build-arg MODEL_NAME=nanbeige \
        --build-arg HF_REPO_NAME=Echo9Zulu/Nanbeige4.1-3B-int4-awq-ov \
        -t ghcr.io/<you>/openarc-nanbeige-model:latest \
        .
    $ docker push ghcr.io/<you>/openarc-nanbeige-model:latest
    ```

    Deployment manifest, now simplified:

    ```yaml
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: &app_name openarc
      labels: &labels
        app: *app_name
    spec:
      selector:
        matchLabels: *labels
      template:
        metadata:
          labels: *labels
        spec:
          containers:
            - name: openarc
              image: ghcr.io/searchsavior/openarc
              env:
                # Must match the model name baked into the model image's openarc_config.json.
                - name: OPENARC_AUTOLOAD_MODEL
                  value: qwen3
                # Point the runtime at the config baked into the model image.
                # The config's relative model_path resolves to /model/<model_name>/.
                - name: OPENARC_CONFIG_FILE
                  value: /model/openarc_config.json
              ports:
                - containerPort: 8000
                  name: http
              # TODO add probes once I fix the /status endpoint bug
              volumeMounts:
                - name: model
                  mountPath: /model
          volumes:
            - name: model
              image:
                reference: ghcr.io/<you>/openarc-qwen3-model
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: &app_name openarc
      labels: &labels
        app: *app_name
    spec:
      selector: *labels
      ports:
        - name: http
          port: 8000
    ```

=== "Fully featured, limited-access HA deployment with model caching"

    If your cluster has multiple _identical_ nodes, you can pre-compile models and cache the result. This allows for making the root filesystem readonly, speeds up model loading time,
    reduces the peak memory requirements (compilation is expensive), and can be re-used across pod restarts on multiple nodes.

    > [!IMPORTANT]
    > Compiled models are dependent on a specific combination of device, kernel, GPU kernel module, libraries, OpenArc, and model versions. A model compiled for an Intel iGPU will not
    > work on an Intel discrete GPU. Any change to any of these dependencies will invalidate the model cache. If you upgrade your node's host OS and deploy a new GPU kernel module (i915
    > driver for example), the cache will need to be rebuilt. For details, see [here](https://docs.openvino.ai/2026/model-server/ovms_docs_model_cache.html).

    OpenArc is stateless. Multiple replicas can be deployed for load balancing or to make OpenArc highly-available.

    Here is an example deployment implementing this. Note that the job will need to complete prior to scaling up the deployment, and will need to be re-ran any time the model needs to be
    recompiled. Helm annotations are added to show what you'd need to do if packaging this in a chart. This requires Kubernetes 1.35+ and a RWX storage class:

    ```yaml
    ---
    apiVersion: resource.k8s.io/v1
    kind: ResourceClaimTemplate
    metadata:
      name: openarc-gpu
      labels:
        app: openarc
    spec:
      spec:
        devices:
          requests:
            - name: gpu
              exactly:
                deviceClassName: gpu.intel.com
                selectors:
                  - cel:
                      # Configure this for your device
                      expression: device.attributes["gpu.intel.com"].family == 'Arc Pro'
    ---
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: openarc-models
      labels:
        app: openarc
      annotations:
        helm.sh/hook: pre-install,pre-upgrade
        helm.sh/hook-weight: "-10"
        helm.sh/resource-policy: keep
    spec:
      # Must be either be RWX, or the deployment must be scaled to 0 prior to running upgrade jobs
      storageClassName: <your-rwx-storage-class>
      accessModes:
        - ReadWriteMany
      resources:
        requests:
          storage: 5Gi
    ---
    # Cache-population Job. Downloads the HF model and runs OpenVINO graph
    # optimization into /models so the runtime Deployment boots into a warm
    # cache and avoids the high-memory cold-compile path. Apply this and
    # wait for completion before applying the Deployment.
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: openarc-setup
      labels:
        app: openarc
      annotations:
        helm.sh/hook: pre-install,pre-upgrade
        helm.sh/hook-weight: "-5"
        helm.sh/hook-delete-policy: before-hook-creation
    spec:
      backoffLimit: 1
      ttlSecondsAfterFinished: 500
      template:
        spec:
          restartPolicy: Never
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            runAsGroup: 1000
            fsGroup: 1000
            fsGroupChangePolicy: OnRootMismatch
          containers:
            - name: openarc
              image: ghcr.io/searchsavior/openarc
              imagePullPolicy: Always
              command: ["python", "-c"]
              # Note: The download logic below can be removed if using an image volume.
              # If adding a netpol, allow access to `huggingface.co`, `**.huggingface.co`, and `**.hf.co` for model download, along with DNS resolution.
              args: 
                - |
                  import os, subprocess, sys, time, urllib.request, urllib.error
                  from huggingface_hub import snapshot_download
                  from pathlib import Path

                  model_name = os.environ["MODEL_NAME"]
                  model_type = os.environ["MODEL_TYPE"]
                  hf_repo = os.environ["HF_REPO"]
                  config_file = os.environ["OPENARC_CONFIG_FILE"]
                  model_path = os.environ.get("MODEL_PATH", Path(model_name) / "model")
                  cache_dir = os.environ.get("CACHE_DIR", Path(model_name) / "cache")
                  engine = os.environ.get("ENGINE", "ovgenai")
                  device = os.environ.get("DEVICE", "GPU")

                  # All paths should be relative to the config file directory
                  os.chdir(Path(config_file).parent)

                  os.makedirs(model_path, exist_ok=True)
                  snapshot_download(repo_id=hf_repo, local_dir=model_path)

                  os.makedirs(os.path.dirname(config_file), exist_ok=True)
                  # Truncate so stale top-level fields from older schemas don't
                  # survive a rerun - `openarc add` only updates a single entry.
                  open(config_file, "w").close()
                  subprocess.run(
                      ["openarc", "add",
                      f"--model-name={model_name}",
                      f"--model-path={model_path}",
                      f"--cache-dir={cache_dir}",
                      f"--engine={engine}",
                      f"--model-type={model_type}",
                      f"--device={device}"],
                      check=True,
                  )

                  os.makedirs(cache_dir, exist_ok=True)

                  # Start the server and wait for it to be ready before trying to load the model
                  server = subprocess.Popen(
                      ["openarc", "serve", "start", "--host", "0.0.0.0", "--port", "8000"],
                      env={**os.environ, "MODEL_NAME": model_name}
                  )
                  try:
                      for i in range(1, 61):
                          try:
                              urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2).read()
                              break
                          except urllib.error.URLError:
                              time.sleep(1)
                      else:
                          sys.exit("Server never became ready within 60s")

                      print(f"Server ready after {i}s; loading {model_name}", flush=True)
                      subprocess.run(["openarc", "load", model_name], check=True)

                      # Verify that the model loaded correctly (and therefore compiled correctly)
                      resp = urllib.request.urlopen("http://localhost:8000/openarc/status", timeout=5).read().decode()
                      if f'"model_name":"{model_name}"' in resp and '"status":"loaded"' in resp:
                          print(f"Model loaded; OV cache populated at {cache_dir}")
                      else:
                          sys.exit(f"Model failed to load: {resp}")
                  finally:
                      server.terminate()
              securityContext:
                readOnlyRootFilesystem: true
                allowPrivilegeEscalation: false
                capabilities:
                  drop:
                    - ALL
              env:
                - name: MODEL_NAME
                  value: qwen3
                - name: MODEL_TYPE
                  value: llm
                - name: HF_REPO
                  value: Echo9Zulu/Qwen3-1.7B-int8_asym-ov
                - name: OPENARC_CONFIG_FILE
                  value: /models/openarc_config.json
                - name: OPENARC_LOG_FILE
                  value: /dev/null
                - name: NUMBA_CACHE_DIR
                  value: /tmp/numba-cache
                # huggingface-cli writes to HF_HOME; the default ~/.cache path
                # is unwritable under readOnlyRootFilesystem.
                - name: HF_HOME
                  value: /tmp/hf-home
              resources:
                requests:
                  cpu: 500m
                  # Compilation requires a significant amount of memory depending on the model. By doing this once,
                  # the peak memory usage for every deployment replica is much lower.
                  memory: &memory 6Gi
                limits:
                  memory: *memory
                claims:
                  - name: gpu
              volumeMounts:
                - name: models
                  mountPath: /models
                - name: tmp
                  mountPath: /tmp
          volumes:
            - name: models
              persistentVolumeClaim:
                claimName: openarc-models
            - name: tmp
              emptyDir: {}
          resourceClaims:
            - name: gpu
              resourceClaimTemplateName: openarc-gpu
    ---
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: &app_name openarc
      labels: &labels
        app: *app_name
    spec:
      replicas: 2 # Optional
      selector:
        matchLabels: *labels
      template:
        metadata:
          labels: *labels
        spec:
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            runAsGroup: 1000
            fsGroup: 1000
            fsGroupChangePolicy: OnRootMismatch
          containers:
            - name: openarc
              image: ghcr.io/searchsavior/openarc
              env:
                - name: OPENARC_AUTOLOAD_MODEL
                  value: qwen3
                - name: OPENARC_CONFIG_FILE
                  value: /models/openarc_config.json
                - name: OPENARC_LOG_FILE
                  value: /dev/null
                - name: NUMBA_CACHE_DIR
                  value: /tmp/numba-cache
              ports:
                - containerPort: 8000
                  name: http
              # TODO add probes once I fix the /status endpoint bug
              volumeMounts:
                - name: models
                  mountPath: /models
                - name: tmp
                  mountPath: /tmp
              securityContext:
                readOnlyRootFilesystem: true
                allowPrivilegeEscalation: false
                capabilities:
                  drop:
                    - ALL
              resources:
                requests:
                  cpu: 100m
                  memory: &memory 2Gi
                limits:
                  memory: *memory
                claims:
                  - name: gpu
          volumes:
            - name: models
              persistentVolumeClaim:
                claimName: openarc-models
            # Note: Audio transcriptions requests for ASR models will store audio contents here for audio files > 1MB.
            - name: tmp
              emptyDir: {}
          resourceClaims:
            - name: gpu
              resourceClaimTemplateName: openarc-gpu
    ---
    apiVersion: policy/v1
    kind: PodDisruptionBudget
    metadata:
      name: &app_name openarc
      labels: &labels
        app: *app_name
    spec:
      minAvailable: 1
      selector:
        matchLabels: *labels
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: &app_name openarc
      labels: &labels
        app: *app_name
    spec:
      selector: *labels
      ports:
        - name: http
          port: 8000
    ```
