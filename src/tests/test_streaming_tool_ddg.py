#!/usr/bin/env python3
"""
Test streaming tool calls with DuckDuckGo search.
Tests proper accumulation of tool call deltas and integration with actual tool execution.
"""
import os
import json
from openai import OpenAI

def ddg_search(query: str) -> list[dict]:
    """Execute a DuckDuckGo search using ddgs library."""
    try:
        import ddgs
        
        print(f"  [SEARCH] Executing: query='{query}'")
        
        with ddgs.DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "position": i,
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", "")
            })
        
        return formatted_results
    
    except Exception as e:
        return [{"error": str(e)}]


def test_streaming_tool_call():
    """Test streaming tool calls with DuckDuckGo search."""
    
    print("="*70)
    print("STREAMING TOOL CALL TEST - DuckDuckGo Search")
    print("="*70)
    
    # Initialize client
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key=os.getenv("OPENARC_API_KEY"),
    )
    
    model_name = "Dolphin-X1"
    
    # Define DuckDuckGo search tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": """
                Search the web. Returns relevant web pages. Use this tool to get information from the internet.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # Initial conversation
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful research assistant with access to web search. If you decide to use a tool, call it first before anything else."
        },
        {
            "role": "user", 
            "content": "What's the most recent OpenVINO version?"
        }
    ]
    
    print("\nStep 1: Streaming request with tool...")
    print(f"Model: {model_name}")
    print("Stream: True")
    print(f"Tools: {tools[0]['function']['name']}")
    
    try:
        # Stream the response
        accumulated_content = ""
        accumulated_tool_calls = {}
        finish_reason = None
        
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            stream=True,
            temperature=0.7
        )
        
        print("\nStreaming chunks:")
        chunk_count = 0
        
        for chunk in stream:
            chunk_count += 1
            choice = chunk.choices[0] if chunk.choices else None
            
            if not choice:
                continue
            
            delta = choice.delta
            finish_reason = choice.finish_reason or finish_reason
            
            # Accumulate content
            if delta.content:
                accumulated_content += delta.content
                print(f"  Chunk {chunk_count}: content='{delta.content[:50]}...'")
            
            # Accumulate tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc_delta.id,
                            "type": tc_delta.type or "function",
                            "function": {
                                "name": "",
                                "arguments": ""
                            }
                        }
                    
                    if tc_delta.id:
                        accumulated_tool_calls[idx]["id"] = tc_delta.id
                    
                    if tc_delta.function:
                        if tc_delta.function.name:
                            accumulated_tool_calls[idx]["function"]["name"] = tc_delta.function.name
                            print(f"  Chunk {chunk_count}: tool_call[{idx}].name='{tc_delta.function.name}'")
                        
                        if tc_delta.function.arguments:
                            accumulated_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments
                            print(f"  Chunk {chunk_count}: tool_call[{idx}].arguments+='{tc_delta.function.arguments[:30]}...'")
        
        print(f"\nTotal chunks received: {chunk_count}")
        print(f"Finish reason: {finish_reason}")
        
        # Check what we accumulated
        if accumulated_tool_calls:
            print("\n✓ Tool calls detected in stream!")
            print(f"Number of tool calls: {len(accumulated_tool_calls)}")
            
            for idx, tc in accumulated_tool_calls.items():
                print(f"\n--- Tool Call {idx} ---")
                print(f"ID: {tc['id']}")
                print(f"Function: {tc['function']['name']}")
                print(f"Arguments: {tc['function']['arguments']}")
                
                # Parse and execute the tool
                try:
                    args = json.loads(tc['function']['arguments'])
                    
                    if tc['function']['name'] == 'search':
                        print("\nStep 2: Executing search...")
                        
                        # Execute search
                        results = ddg_search(
                            query=args['query']
                        )
                        
                        print(f"  [SEARCH] Got {len(results)} results")
                        
                        # Add tool call to conversation
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tc['id'],
                                "type": "function",
                                "function": {
                                    "name": tc['function']['name'],
                                    "arguments": tc['function']['arguments']
                                }
                            }]
                        })
                        
                        # Add tool results
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc['id'],
                            "content": json.dumps(results, indent=2)
                        })
                        
                        print("\nStep 3: Getting final response from model...")
                        final_response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            stream=False  
                        )
                        
                        final_text = final_response.choices[0].message.content
                        print("\n--- Model's Final Response ---")
                        print(final_text)
                        print("--- End Response ---")
                        
                        # Validate results
                        print("\n" + "="*70)
                        if results and not results[0].get("error"):
                            print("✓ TEST PASSED - Streaming tool call workflow successful!")
                            print("  - Streamed tool call detected")
                            print("  - Search executed successfully")
                            print(f"  - Got {len(results)} search results")
                            print("  - Model provided summary")
                            return True
                        else:
                            print("✗ TEST FAILED - Search execution failed")
                            return False
                    
                except json.JSONDecodeError as e:
                    print(f"\n✗ Failed to parse tool arguments: {e}")
                    print(f"Raw arguments: {tc['function']['arguments']}")
                    return False
        
        elif accumulated_content:
            print("\n✗ No tool calls detected in stream")
            print(f"Accumulated content: {accumulated_content[:200]}")
            print("\nThe model responded with text instead of calling the tool.")
            return False
        
        else:
            print("\n✗ Empty response")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_streaming_tool_call()
    exit(0 if success else 1)

