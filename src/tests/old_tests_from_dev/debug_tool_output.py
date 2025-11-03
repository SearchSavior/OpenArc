#!/usr/bin/env python3
"""
Debug script to see what the model actually outputs when given tools.
Run this to diagnose tool calling issues.
"""
import os
import json
from openai import OpenAI

def debug_tool_output():
    """Check what the model outputs with and without tools."""
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key=os.getenv("OPENARC_API_KEY"),
    )
    
    model_name = "Dolphin-X1"
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write_word",
                "description": "Writes a specific word to output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word": {
                            "type": "string",
                            "description": "The word to write"
                        }
                    },
                    "required": ["word"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Always format tools with xml tags like <tool_call>JSON</tool_call>"},
        {"role": "user", "content": "Please use the write_word tool to write the word 'pirate'."}
    ]
    
    print("="*70)
    print("DEBUG: Raw model output with tools")
    print("="*70)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            temperature=0.1,  # Lower temperature for more consistent output
        )
        
        msg = response.choices[0].message
        
        print(f"\nFinish reason: {response.choices[0].finish_reason}")
        print(f"\nMessage role: {msg.role}")
        print(f"\nMessage content:")
        print(msg.content)
        print(f"\nMessage tool_calls:")
        print(msg.tool_calls)
        
        if msg.tool_calls:
            print("\n✓ SUCCESS: Tool calls detected by server!")
            for i, tc in enumerate(msg.tool_calls):
                print(f"\nTool call {i+1}:")
                print(f"  ID: {tc.id}")
                print(f"  Function: {tc.function.name}")
                print(f"  Arguments: {tc.function.arguments}")
        else:
            print("\n✗ ISSUE: No tool calls detected")
            print("\nThe model output above should contain tool call markers.")
            print("Check your model's chat template and training.")
            print("\nExpected formats:")
            print("  1. <tool_call>{\"name\": \"write_word\", \"arguments\": {\"word\": \"pirate\"}}</tool_call>")
            print("  2. <|python_tag|>{\"name\": \"write_word\", \"arguments\": {\"word\": \"pirate\"}}<|eom_id|>")
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tool_output()

