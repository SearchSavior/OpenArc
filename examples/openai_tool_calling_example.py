"""
Example implementation of OpenAI-compatible tool calling server-side processing.

This demonstrates the key server-side post-processing steps needed for proper
OpenAI-compatible function/tool calling support.
"""

import json
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """Represents a single tool call."""
    id: str
    type: str = "function"
    function: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function
        }


@dataclass
class StreamingToolCallState:
    """Tracks state for accumulating streaming tool call chunks."""
    tool_calls: Dict[int, ToolCall] = field(default_factory=dict)
    
    def update(self, index: int, delta: Dict[str, Any]) -> ToolCall:
        """Update tool call at index with delta information."""
        if index not in self.tool_calls:
            # Initialize new tool call
            tool_id = delta.get("id")
            if not tool_id:
                tool_id = generate_tool_call_id()
            
            self.tool_calls[index] = ToolCall(
                id=tool_id,
                type=delta.get("type", "function"),
                function={
                    "name": delta.get("function", {}).get("name", ""),
                    "arguments": ""
                }
            )
        
        # Accumulate function name if provided
        if delta.get("function", {}).get("name"):
            self.tool_calls[index].function["name"] = delta["function"]["name"]
        
        # Accumulate arguments incrementally
        if delta.get("function", {}).get("arguments"):
            self.tool_calls[index].function["arguments"] += delta["function"]["arguments"]
        
        return self.tool_calls[index]
    
    def get_all_tool_calls(self) -> List[Dict]:
        """Get all tool calls as dictionaries."""
        return [tc.to_dict() for tc in sorted(self.tool_calls.values(), key=lambda x: list(self.tool_calls.keys())[list(self.tool_calls.values()).index(x)])]


def generate_tool_call_id() -> str:
    """
    Generate a unique tool call ID in OpenAI format.
    
    Returns:
        String in format 'call_<random_hex>' (e.g., 'call_abc123def456')
    """
    return f"call_{uuid.uuid4().hex[:24]}"


def parse_model_tool_output(model_output: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse model output to extract tool calls.
    
    This is MODEL-SPECIFIC and depends on how your model outputs tool calls.
    Common formats:
    
    1. Special tokens: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. JSON: {"tool_calls": [{"name": "...", "arguments": {...}}]}
    3. XML: <tool_call><name>...</name><arguments>...</arguments></tool_call>
    
    Args:
        model_output: Raw output string from the model
        
    Returns:
        List of tool call dictionaries, or None if no tools detected
        
    Example output:
        [
            {
                "name": "get_weather",
                "arguments": {"location": "San Francisco", "unit": "celsius"}
            }
        ]
    """
    # Example implementation for JSON format
    # REPLACE THIS with your actual parsing logic!
    
    # Try to find tool call markers (example format)
    if "<tool_call>" in model_output:
        # Parse special token format
        tool_calls = []
        import re
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, model_output, re.DOTALL)
        
        for match in matches:
            try:
                tool_data = json.loads(match)
                tool_calls.append({
                    "name": tool_data.get("name"),
                    "arguments": tool_data.get("arguments", {})
                })
            except json.JSONDecodeError:
                # Model generated invalid JSON - still include it
                # (validation is client's responsibility)
                tool_calls.append({
                    "name": "unknown",
                    "arguments": match
                })
        
        return tool_calls if tool_calls else None
    
    # Try JSON format
    try:
        data = json.loads(model_output)
        if "tool_calls" in data:
            return data["tool_calls"]
    except json.JSONDecodeError:
        pass
    
    return None


def format_tool_call_response(
    tool_calls: List[Dict[str, Any]],
    content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format parsed tool calls into OpenAI-compatible response structure.
    
    Args:
        tool_calls: List of tool call dicts with 'name' and 'arguments'
        content: Optional text content (can coexist with tool calls)
        
    Returns:
        OpenAI-compatible message dictionary
        
    Example:
        >>> tool_calls = [{"name": "get_weather", "arguments": {"location": "SF"}}]
        >>> format_tool_call_response(tool_calls)
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123...",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "SF"}'
                    }
                }
            ]
        }
    """
    formatted_tool_calls = []
    
    for tc in tool_calls:
        # Ensure arguments is a JSON string
        arguments = tc.get("arguments", {})
        if isinstance(arguments, dict):
            arguments_str = json.dumps(arguments)
        else:
            # Already a string (possibly invalid JSON - that's OK)
            arguments_str = str(arguments)
        
        formatted_tool_calls.append({
            "id": generate_tool_call_id(),
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": arguments_str
            }
        })
    
    return {
        "role": "assistant",
        "content": content,  # Often None when tool_calls present
        "tool_calls": formatted_tool_calls
    }


def create_non_streaming_response(
    message: Dict[str, Any],
    finish_reason: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0
) -> Dict[str, Any]:
    """
    Create full OpenAI-compatible chat completion response.
    
    Args:
        message: Message dict (from format_tool_call_response or text message)
        finish_reason: "stop", "length", "tool_calls", or "content_filter"
        model: Model name/identifier
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        
    Returns:
        Complete OpenAI chat completion response
    """
    import time
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


def create_streaming_chunk(
    delta: Dict[str, Any],
    finish_reason: Optional[str],
    model: str,
    chunk_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a streaming chunk in OpenAI format.
    
    Args:
        delta: Delta dictionary (incremental changes)
        finish_reason: Finish reason (None for intermediate chunks)
        model: Model name
        chunk_id: Optional chunk ID (reuse same ID for all chunks in stream)
        
    Returns:
        OpenAI-compatible streaming chunk
    """
    import time
    
    if chunk_id is None:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None
            }
        ]
    }


# Example usage for NON-STREAMING
def example_non_streaming():
    """Example of processing a non-streaming tool call response."""
    
    # 1. Model generates output (this is model-specific)
    model_output = """<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}</tool_call>"""
    
    # 2. Parse tool calls from model output
    tool_calls = parse_model_tool_output(model_output)
    
    if tool_calls:
        # 3. Format into OpenAI structure
        message = format_tool_call_response(tool_calls)
        
        # 4. Create complete response
        response = create_non_streaming_response(
            message=message,
            finish_reason="tool_calls",
            model="my-model-v1",
            prompt_tokens=50,
            completion_tokens=25
        )
        
        print("Non-streaming response:")
        print(json.dumps(response, indent=2))


# Example usage for STREAMING
def example_streaming():
    """Example of processing streaming tool call chunks."""
    
    # Simulated model output chunks
    model_chunks = [
        {"role": "assistant"},  # Initial chunk
        {"tool_calls": [{"index": 0, "id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": 'ation": '}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"San Francis'}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": 'co"}'}}]},
        {"finish_reason": "tool_calls"},  # Final chunk
    ]
    
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    state = StreamingToolCallState()
    
    print("\nStreaming chunks:")
    for i, model_chunk in enumerate(model_chunks):
        # Process tool calls in chunk
        if "tool_calls" in model_chunk:
            for tc_delta in model_chunk["tool_calls"]:
                state.update(tc_delta["index"], tc_delta)
        
        # Create OpenAI-compatible chunk
        delta = {}
        if "role" in model_chunk:
            delta["role"] = model_chunk["role"]
        if "tool_calls" in model_chunk:
            delta["tool_calls"] = model_chunk["tool_calls"]
        
        finish_reason = model_chunk.get("finish_reason")
        
        chunk = create_streaming_chunk(
            delta=delta,
            finish_reason=finish_reason,
            model="my-model-v1",
            chunk_id=chunk_id
        )
        
        print(f"\nChunk {i+1}:")
        print(json.dumps(chunk, indent=2))
    
    # Show accumulated result
    print("\n\nAccumulated tool calls:")
    print(json.dumps(state.get_all_tool_calls(), indent=2))


# Example with MULTIPLE PARALLEL tool calls
def example_parallel_tools():
    """Example of handling multiple parallel tool calls."""
    
    model_output = """
    <tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}}</tool_call>
    <tool_call>{"name": "get_time", "arguments": {"timezone": "America/Los_Angeles"}}</tool_call>
    """
    
    tool_calls = parse_model_tool_output(model_output)
    
    if tool_calls:
        message = format_tool_call_response(tool_calls)
        response = create_non_streaming_response(
            message=message,
            finish_reason="tool_calls",
            model="my-model-v1"
        )
        
        print("\n\nParallel tool calls response:")
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    print("=" * 70)
    print("OpenAI Tool Calling Server-Side Processing Examples")
    print("=" * 70)
    
    example_non_streaming()
    print("\n" + "=" * 70 + "\n")
    
    example_streaming()
    print("\n" + "=" * 70 + "\n")
    
    example_parallel_tools()

