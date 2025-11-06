import os
from huggingface_hub import HfApi

hf_api_key = os.getenv("HF_TOKEN")

def search_huggingface(query: str, search_type: str = "model", limit: int = 10):
    """
    Search the Hugging Face Hub for models or datasets.
    
    Args:
        query: Search query string
        search_type: Either "model" or "dataset"
        limit: Maximum number of results to return
    
    Returns:
        List of search results with metadata
    """
    api = HfApi(token=hf_api_key)
    
    if search_type == "model":
        results = api.list_models(search=query, limit=limit)
    elif search_type == "dataset":
        results = api.list_datasets(search=query, limit=limit)
    else:
        raise ValueError("search_type must be 'model' or 'dataset'")
    
    return list(results)


def main():
    """Main entrypoint for the HF explorer CLI."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hf_tools.py <query> [search_type] [limit]")
        print("  search_type: 'model' (default) or 'dataset'")
        print("  limit: number of results (default: 10)")
        sys.exit(1)
    
    query = sys.argv[1]
    search_type = sys.argv[2] if len(sys.argv) > 2 else "model"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    results = search_huggingface(query, search_type, limit)
    
    print(f"\nSearch results for '{query}' ({search_type}s):\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.id}")
        if hasattr(result, 'downloads'):
            print(f"   Downloads: {result.downloads}")
        print()


if __name__ == "__main__":
    main()