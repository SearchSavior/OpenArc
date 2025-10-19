"""
Maintaining many HF repos is hard. 
This script is the example I have been using to download from repos
ending in -OpenVINO which contain many different quantizations.

Othewise you get the whole repo which can be very large.
""" 
from huggingface_hub import snapshot_download

repo_id = "Echo9Zulu/Phi-lthy4-OpenVINO"     

# Choose the weights you want
repo_directory = "Phi-lthy4-int4_sym-awq-ov"

# Where you want to save it
local_dir = "./Echo9Zulu_Phi-lthy4-OpenVINO/Phi-lthy4-int4_sym-awq-ov"

snapshot_download(
    repo_id=repo_id,
    allow_patterns=[f"{repo_directory}/*"], 
    local_dir=local_dir,
    local_dir_use_symlinks=True
) 

print("Download complete!")


