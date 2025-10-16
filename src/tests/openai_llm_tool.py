import os
import json
from openai import OpenAI


def test_tool_calling():
    """Test tool calling with Qwen3-4B-2507 model - runs 50 times."""
    
    # Initialize client
    try:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key=os.getenv("OPENARC_API_KEY"),
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    model_name = "qwen25-14b"
    
    # Track results
    total_tests = 50
    successes = 0
    failures = 0
    
    # Define a simple tool for writing a word
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
    
    print(f"Running {total_tests} tool calling tests...\n")
    
    for test_num in range(1, total_tests + 1):
        print(f"{'='*60}")
        print(f"Test {test_num}/{total_tests}")
        print(f"{'='*60}")
        
        # Initial conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "Please use the write_word tool to write the word 'pirate'."}
        ]
        
        try:
            # First request - model should call the tool
            print("Step 1: Sending request with tools...")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                #tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            # Check if model made a tool call
            if assistant_message.tool_calls:
                print("✓ Tool call detected!")
                tool_call = assistant_message.tool_calls[0]
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")
                
                # Parse the arguments
                try:
                    args = json.loads(tool_call.function.arguments)
                    word_written = args.get("word", "")
                    
                    # Add the assistant's tool call to conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        ]
                    })
                    
                    # Add the tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Successfully wrote: {word_written}"
                    })
                    
                    # Second request - get final response
                    print("Step 2: Sending tool result back to model...")
                    final_response = client.chat.completions.create(
                        model=model_name,
                        messages=messages
                    )
                    
                    final_text = final_response.choices[0].message.content
                    print(f"Model final response: {final_text}")
                    
                    # Check if the response contains the expected phrase
                    if word_written.lower() == "pirate" and "pirate" in final_text.lower():
                        print("✓ Test passed! LLM wrote pirate. Tool calling implementation confirmed\n")
                        successes += 1
                    else:
                        print(f"✗ Test failed. Expected 'pirate' but got: {word_written}\n")
                        failures += 1
                        
                except json.JSONDecodeError as e:
                    print(f"✗ Failed to parse tool arguments: {e}\n")
                    failures += 1
            else:
                print("✗ No tool calls detected in response")
                if assistant_message.content:
                    print(f"Model response: {assistant_message.content}")
                print()
                failures += 1
                
        except Exception as e:
            print(f"✗ Error during tool calling test: {e}\n")
            failures += 1
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Successes: {successes} ({successes/total_tests*100:.1f}%)")
    print(f"Failures: {failures} ({failures/total_tests*100:.1f}%)")
    print(f"{'='*60}")
    
    return successes == total_tests


if __name__ == "__main__":
    test_tool_calling()

