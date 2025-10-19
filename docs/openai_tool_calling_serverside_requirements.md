# OpenAI-Compatible Tool Calling: Server-Side Requirements

This document outlines the concrete server-side post-processing requirements for implementing OpenAI-compatible function/tool calling based on the official OpenAI Python library.

## 1. Request Structure

### Tool Definition (Input)
```python
# Request payload includes tools array
{
    "model": "model-name",
    "messages": [...],
    "tools": [
        {
            "type": "function",  # Currently only "function" is supported
            "function": {
                "name": "function_name",  # Required: a-z, A-Z, 0-9, underscores, dashes (max 64 chars)
                "description": "What the function does",  # Optional but recommended
                "parameters": {  # JSON Schema object
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "..."},
                        "param2": {"type": "number"}
                    },
                    "required": ["param1"]
                },
                "strict": false  # Optional: enable strict schema adherence
            }
        }
    ],
    "tool_choice": "auto"  # "auto" | "none" | "required" | {"type": "function", "function": {"name": "..."}}
}
```

### Tool Choice Parameter
- `"none"`: Model will NOT call any tool
- `"auto"`: Model can choose to call tools or generate text (default when tools present)
- `"required"`: Model MUST call at least one tool
- `{"type": "function", "function": {"name": "my_function"}}`: Force specific tool

### Parallel Tool Calls
- `parallel_tool_calls`: boolean (default: true)
- Controls whether multiple tools can be called simultaneously

## 2. Response Structure

### Non-Streaming Response

```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,  # null when tool_calls present, or string for text
                "tool_calls": [
                    {
                        "id": "call_abc123",  # Unique ID for this tool call
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"San Francisco\"}"  # JSON string
                        }
                    }
                ],
                "refusal": null  # Present if model refuses
            },
            "finish_reason": "tool_calls",  # "stop" | "length" | "tool_calls" | "content_filter"
            "logprobs": null
        }
    ],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "total_tokens": 70
    }
}
```

### Streaming Response

Streaming returns `ChatCompletionChunk` objects with deltas:

```python
# Chunk 1: Initial chunk with role
{
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": null,
                "tool_calls": null
            },
            "finish_reason": null
        }
    ]
}

# Chunk 2: Tool call ID and function name
{
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
        {
            "index": 0,
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,  # Index of this tool call
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": ""
                        }
                    }
                ]
            },
            "finish_reason": null
        }
    ]
}

# Chunks 3-N: Arguments streamed incrementally
{
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
        {
            "index": 0,
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {
                            "arguments": "{\"loc"  # Partial JSON string
                        }
                    }
                ]
            },
            "finish_reason": null
        }
    ]
}

# Final chunk
{
    "id": "chatcmpl-123",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "model-name",
    "choices": [
        {
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls"
        }
    ]
}
```

## 3. Server-Side Processing Requirements

### 3.1 Tool Call ID Generation
- **MUST** generate unique IDs for each tool call (e.g., `"call_abc123"`)
- Format: typically `"call_"` prefix + random alphanumeric string
- IDs persist across streaming chunks for the same tool call

### 3.2 Finish Reason Logic
```python
finish_reason_map = {
    "tool_calls": "Model decided to call tool(s)",
    "stop": "Natural stop or hit stop sequence",
    "length": "Reached max_tokens limit",
    "content_filter": "Content filtered",
    "function_call": "Deprecated - use tool_calls instead"
}
```

### 3.3 Streaming Accumulation Logic

The server must implement proper accumulation of streaming chunks:

```python
# Pseudo-code for accumulation
def accumulate_tool_call_chunks(chunks):
    tool_calls = {}  # index -> accumulated tool call
    
    for chunk in chunks:
        for choice in chunk.choices:
            for tool_call_delta in choice.delta.tool_calls or []:
                idx = tool_call_delta.index
                
                if idx not in tool_calls:
                    # Initialize new tool call
                    tool_calls[idx] = {
                        "id": tool_call_delta.id,
                        "type": tool_call_delta.type,
                        "function": {
                            "name": tool_call_delta.function.name or "",
                            "arguments": tool_call_delta.function.arguments or ""
                        }
                    }
                else:
                    # Accumulate arguments
                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tool_call_delta.function.arguments
    
    return list(tool_calls.values())
```

### 3.4 Message Content vs Tool Calls
- When `tool_calls` is present, `content` is typically `null`
- Model may return both `content` and `tool_calls` in some cases
- Server should preserve both fields in response

### 3.5 Multiple Tool Calls (Parallel)
When `parallel_tool_calls=true`:
```python
{
    "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
            {
                "index": 0,  # First tool call
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "..."}
            },
            {
                "index": 1,  # Second tool call
                "id": "call_def456", 
                "type": "function",
                "function": {"name": "get_time", "arguments": "..."}
            }
        ]
    }
}
```

### 3.6 JSON Arguments Validation
**IMPORTANT NOTE from OpenAI docs:**
> "Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function."

Server-side considerations:
- Arguments are returned as JSON **strings**, not objects
- Server does NOT need to validate JSON correctness
- Server does NOT need to validate against schema
- Validation is client's responsibility
- However, proper tokenization can help models generate valid JSON

### 3.7 Deprecated Function Call Format
Legacy format (still supported):
```python
{
    "message": {
        "function_call": {
            "name": "get_weather",
            "arguments": "{\"location\": \"SF\"}"
        }
    },
    "finish_reason": "function_call"
}
```

## 4. Tokenization Requirements

### Chat Template with Tools
For models to properly output tool calls, the tokenizer/chat template must:

1. **Accept tools in template**: Format tools into the prompt
2. **Generate structured output**: Model should output in a parseable format
3. **Support tool tokens**: Special tokens for tool call boundaries

Example flow:
```
User prompt + Tool definitions (via chat template)
    ↓
Model generates: <tool_call>{"name": "get_weather", "arguments": {"location": "SF"}}</tool_call>
    ↓
Server post-processes: Extract, format into OpenAI structure, assign ID
```

### Common Approaches:

#### Approach 1: Special Tokens
```
<|tool_call|>function_name\n{json_args}<|end_tool_call|>
```

#### Approach 2: JSON Mode
```
Model output: {"tool_calls": [{"name": "...", "arguments": {...}}]}
Server: Parse and restructure
```

#### Approach 3: XML/Markdown
```
<tool_call>
name: function_name
arguments: {"param": "value"}
</tool_call>
```

## 5. Implementation Checklist

### Request Processing
- [ ] Parse `tools` array from request
- [ ] Parse `tool_choice` parameter
- [ ] Parse `parallel_tool_calls` parameter
- [ ] Format tools into model prompt via chat template

### Response Generation (Non-Streaming)
- [ ] Detect tool call intent from model output
- [ ] Parse tool call(s) from model output
- [ ] Generate unique IDs for each tool call (e.g., `call_` + random string)
- [ ] Format into OpenAI structure with `tool_calls` array
- [ ] Set `finish_reason` to `"tool_calls"`
- [ ] Set `content` appropriately (null or text)

### Response Generation (Streaming)
- [ ] Initial chunk: Send role
- [ ] Tool call start: Send tool call with `id`, `type`, `function.name`, and empty `arguments`
- [ ] Argument chunks: Stream `arguments` incrementally with correct `index`
- [ ] Handle multiple tool calls with different indices
- [ ] Final chunk: Send `finish_reason`
- [ ] Track tool call indices properly

### Edge Cases
- [ ] Handle empty tool calls list
- [ ] Handle invalid JSON from model (pass through as-is)
- [ ] Handle mixed content + tool_calls responses
- [ ] Handle tool_choice="required" enforcement
- [ ] Handle tool_choice with specific function
- [ ] Support legacy function_call format if needed

## 6. Example Server-Side Code Structure

```python
import json
import uuid
from typing import List, Dict, Optional

def generate_tool_call_id() -> str:
    """Generate unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:24]}"

def parse_model_output_for_tools(model_output: str, tools: List[Dict]) -> Optional[List[Dict]]:
    """
    Parse model output to extract tool calls.
    Implementation depends on your model's output format.
    """
    # Example: Parse from special tokens or JSON
    # This is model-specific!
    pass

def format_tool_call_response(tool_calls: List[Dict]) -> Dict:
    """Format tool calls into OpenAI-compatible response."""
    formatted_tool_calls = []
    
    for idx, tc in enumerate(tool_calls):
        formatted_tool_calls.append({
            "id": generate_tool_call_id(),
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"]
            }
        })
    
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": formatted_tool_calls
    }

def accumulate_streaming_tool_calls(state: Dict, chunk_delta: Dict) -> Dict:
    """Accumulate tool call deltas from streaming chunks."""
    if not chunk_delta.get("tool_calls"):
        return state
    
    if "tool_calls" not in state:
        state["tool_calls"] = {}
    
    for tc_delta in chunk_delta["tool_calls"]:
        idx = tc_delta["index"]
        
        if idx not in state["tool_calls"]:
            state["tool_calls"][idx] = {
                "id": tc_delta.get("id"),
                "type": tc_delta.get("type"),
                "function": {
                    "name": tc_delta.get("function", {}).get("name", ""),
                    "arguments": ""
                }
            }
        
        # Accumulate arguments
        if tc_delta.get("function", {}).get("arguments"):
            state["tool_calls"][idx]["function"]["arguments"] += tc_delta["function"]["arguments"]
    
    return state
```

## 7. Testing Recommendations

Test cases to implement:

1. **Single tool call** - Basic case
2. **Multiple parallel tool calls** - Test indexing
3. **Tool call with complex JSON** - Nested objects, arrays
4. **Tool call with invalid JSON** - Ensure pass-through
5. **Mixed content and tool calls** - Some models may output both
6. **Streaming accumulation** - Verify correct reconstruction
7. **Tool choice enforcement** - Test "required", "none", specific function
8. **Empty tools list** - Model should behave normally
9. **Unicode in arguments** - International characters
10. **Large argument strings** - Test chunking in streaming

## 8. Key Differences from Text Generation

| Aspect | Text Generation | Tool Calling |
|--------|----------------|--------------|
| finish_reason | "stop" or "length" | "tool_calls" |
| message.content | Always present | Often null |
| message.tool_calls | null | Array of tool calls |
| ID generation | Not needed | Required per tool call |
| Streaming deltas | Content strings | Tool call objects with indices |
| Validation | N/A | Client-side responsibility |

## References

Based on analysis of:
- `openai` Python library v1.x
- `/types/chat/chat_completion_message_tool_call.py`
- `/types/chat/chat_completion_chunk.py`
- `/lib/streaming/chat/_completions.py`
- OpenAI API documentation

---

**Important Notes:**

1. **Tool call IDs must be unique** - Generate them server-side
2. **Arguments are JSON strings** - Not parsed objects
3. **Streaming uses indices** - Track tool calls by index during streaming
4. **finish_reason matters** - Set to "tool_calls" when tools are invoked
5. **Validation is client-side** - Server just formats the output correctly

