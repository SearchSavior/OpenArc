"""
Integration test for the /v1/rerank endpoint

This test sends a query and three documents to the rerank endpoint
and validates the response structure.
"""

import os
import requests


def test_rerank_endpoint():
    """Test the /v1/rerank endpoint with three documents"""
    
    # Configuration
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/v1/rerank"
    
    # Get API key from environment variable
    api_key = os.getenv("OPENARC_API_KEY")
    if not api_key:
        raise ValueError("OPENARC_API_KEY environment variable is not set")
    
    # Set up headers with Bearer token authentication
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Prepare test data with query and three documents
    payload = {
        "model": "Qwen3-Rerank",
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital and most populous city of France.",
            "Berlin is the capital and largest city of Germany.",
            "London is the capital and largest city of England and the United Kingdom."
        ]
    }
    
    # Send POST request to the rerank endpoint
    print(f"Sending request to {endpoint}...")
    response = requests.post(endpoint, json=payload, headers=headers)
    
    # Print response details
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Validate response
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    
    result = response.json()
    
    # Validate response structure
    assert "id" in result, "Response missing 'id' field"
    assert "object" in result, "Response missing 'object' field"
    assert "created" in result, "Response missing 'created' field"
    assert "model" in result, "Response missing 'model' field"
    assert "data" in result, "Response missing 'data' field"
    assert "usage" in result, "Response missing 'usage' field"
    
    # Validate response values
    assert result["object"] == "list", f"Expected object='list', got {result['object']}"
    assert result["model"] == "Qwen3-Rerank", f"Expected model='Qwen3-Rerank', got {result['model']}"
    
    # Validate data structure
    data = result["data"]
    assert isinstance(data, list), "Expected 'data' to be a list"
    assert len(data) == 3, f"Expected 3 ranked documents, got {len(data)}"
    
    # Validate each document in the response
    for i, doc in enumerate(data):
        assert "index" in doc, f"Document {i} missing 'index' field"
        assert "object" in doc, f"Document {i} missing 'object' field"
        assert "ranked_documents" in doc, f"Document {i} missing 'ranked_documents' field"
        assert doc["index"] == i, f"Expected index {i}, got {doc['index']}"
        assert doc["object"] == "ranked_documents", f"Expected object='ranked_documents', got {doc['object']}"
    
    # Validate usage information
    usage = result["usage"]
    assert "prompt_tokens" in usage, "Usage missing 'prompt_tokens' field"
    assert "total_tokens" in usage, "Usage missing 'total_tokens' field"
    assert isinstance(usage["prompt_tokens"], int), "prompt_tokens should be an integer"
    assert isinstance(usage["total_tokens"], int), "total_tokens should be an integer"
    
    print("✓ All assertions passed!")
    print(f"Successfully reranked {len(data)} documents")
    
    return result


if __name__ == "__main__":
    try:
        result = test_rerank_endpoint()
        print("\n✓ Test completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise

