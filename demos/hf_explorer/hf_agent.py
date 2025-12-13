import os
import sys
import json
from openai import OpenAI
from hf_tools import search_huggingface

# Initialize OpenAI client with OpenArc API
client = OpenAI(
    api_key=os.getenv("OPENARC_API_KEY"),
    base_url="http://localhost:8000/v1"
)

MODEL = "Qwen3-4B-2507"

# Define the tool schema for OpenAI function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_huggingface",
            "description": "Search the Hugging Face Hub for models or datasets. Returns a list of matching items with metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string to find models or datasets"
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["model", "dataset"],
                        "description": "Whether to search for models or datasets",
                        "default": "model"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def execute_tool_call(tool_call):
    """Execute a tool call and return the result."""
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    if function_name == "search_huggingface":
        results = search_huggingface(**arguments)
        # Format results for the LLM
        formatted_results = []
        for result in results:
            result_dict = {
                "id": result.id,
            }
            if hasattr(result, 'downloads'):
                result_dict['downloads'] = result.downloads
            if hasattr(result, 'likes'):
                result_dict['likes'] = result.likes
            if hasattr(result, 'tags'):
                result_dict['tags'] = result.tags[:5] if result.tags else []  # Limit tags
            formatted_results.append(result_dict)
        
        return json.dumps(formatted_results, indent=2)
    
    return json.dumps({"error": "Unknown function"})


def chat_loop():
    """Interactive chat loop with the agent."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can search the Hugging Face Hub for models and datasets. When users ask about models or datasets, use the search_huggingface function to find relevant results."
        }
    ]
    
    print("HuggingFace Explorer Agent")
    print("=" * 50)
    print("Ask me to search for models or datasets on HuggingFace!")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Get response from LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        # Check if the model wants to call a function
        if response_message.tool_calls:
            # Execute all tool calls
            for tool_call in response_message.tool_calls:
                print(f"\n[Calling: {tool_call.function.name}]")
                result = execute_tool_call(tool_call)
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            # Get final response after tool execution
            final_response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            final_message = final_response.choices[0].message
            messages.append(final_message)
            print(f"\nAssistant: {final_message.content}\n")
        else:
            # Direct response without tool call
            print(f"\nAssistant: {response_message.content}\n")


def main():
    """Main entrypoint."""
    try:
        chat_loop()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

