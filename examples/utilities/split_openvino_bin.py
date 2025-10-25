import os
import json
import argparse

def split_bin_file(
    model_dir: str,
    chunk_size_gb: int = 1
):
    """
    Split a large OpenVINO .bin file into N-GB chunks with an index.
    """
    bin_path = os.path.join(model_dir, "openvino_model.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError("openvino_model.bin not found")
    
    output_dir = os.path.join(model_dir, "bin_chunks")
    os.makedirs(output_dir, exist_ok=True)

    chunk_size = chunk_size_gb * 1024**3
    base_name = "openvino_model.bin"

    index = {
        "original_file": bin_path,
        "original_size": os.path.getsize(bin_path),
        "chunk_size": chunk_size,
        "chunks": []
    }

    with open(bin_path, "rb") as f:
        i = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            i += 1
            part_name = f"{base_name}.part{i:03d}"
            part_path = os.path.join(output_dir, part_name)

            with open(part_path, "wb") as pf:
                pf.write(chunk)

            index["chunks"].append({
                "part": part_name,
                "size": len(chunk)
            })

    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as jf:
        json.dump(index, jf, indent=2)

    print(f"Split complete. {len(index['chunks'])} chunks created.")

def reassemble_bin_file(chunks_dir: str, output_path: str = None):
    """
    Reassemble a split OpenVINO .bin file from chunks using the index.
    """
    index_path = os.path.join(chunks_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError("index.json not found")
    
    with open(index_path, "r") as jf:
        index = json.load(jf)
    
    if output_path is None:
        output_path = os.path.join(chunks_dir, "openvino_model_reassembled.bin")
    
    with open(output_path, "wb") as output_file:
        for chunk_info in index["chunks"]:
            part_path = os.path.join(chunks_dir, chunk_info["part"])
            
            if not os.path.exists(part_path):
                raise FileNotFoundError(f"Chunk not found: {chunk_info['part']}")
            
            with open(part_path, "rb") as part_file:
                chunk_data = part_file.read()
                
                if len(chunk_data) != chunk_info["size"]:
                    raise ValueError(f"Chunk size mismatch: {chunk_info['part']}")
                
                output_file.write(chunk_data)
    
    final_size = os.path.getsize(output_path)
    if final_size != index["original_size"]:
        raise ValueError("File size mismatch after reassembly")
    
    print(f"Reassembly complete. File saved: {os.path.basename(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="Split or reassemble OpenVINO .bin files")
    
    parser.add_argument("--model-dir", nargs="?", help="OpenVINO model directory (default: current directory)")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split-chunk-size", type=int, metavar="GB", help="Split into GB chunks")
    group.add_argument("--assemble", action="store_true", help="Reassemble chunks")
    
    args = parser.parse_args()
    
    # Use provided directory or current directory
    target_dir = args.model_dir if args.model_dir else os.getcwd()
    target_dir = os.path.abspath(target_dir)
    
    if args.split_chunk_size:
        if args.split_chunk_size <= 0:
            parser.error("Chunk size must be > 0")
        split_bin_file(target_dir, args.split_chunk_size)
        
    elif args.assemble:
        chunks_dir = os.path.join(target_dir, "bin_chunks")
        if not os.path.exists(chunks_dir):
            parser.error("bin_chunks directory not found")
        reassemble_bin_file(chunks_dir)

if __name__ == "__main__":
    main()
