import os
from openai import OpenAI
from datetime import datetime

def list_openai_models():
    """List all available OpenAI models"""
    
    # Initialize client
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=os.getenv("OPENARC_API_KEY")
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return
    
    try:
        # Get all models
        response = client.models.list()
        models = response.data
        
        print(f"Found {len(models)} models:\n")
        
        # Sort models by name
        sorted_models = sorted(models, key=lambda x: x.id)
        
        for model in sorted_models:
            created_date = datetime.fromtimestamp(model.created).strftime("%Y-%m-%d")
            print(f"Model: {model.id}")
            print(f"  Created: {created_date}")
            print(f"  Owner: {model.owned_by}")
            print()
            
    except Exception as e:
        print(f"Error fetching models: {e}")

if __name__ == "__main__":
    list_openai_models()