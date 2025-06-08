from huggingface_hub import snapshot_download

repo_id = "Echo9Zulu/gemma-3-4b-it-OpenVINO"     

# Choose the weights you want
repo_directory = "gemma-3-4b-it-int8_asym-ov"

# Where you want to save it
local_dir = "./Echo9Zulu_gemma-3-4b-it-OpenVINO"

snapshot_download(
    repo_id=repo_id,
    allow_patterns=[f"{repo_directory}/*"], 
    local_dir=local_dir,
    local_dir_use_symlinks=True
) 

print("Download complete!")
