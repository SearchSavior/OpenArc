from smolagents import ToolCallingAgent, DuckDuckGoSearchTool
from smolagents.models import OpenAIServerModel
import os

api_key = os.getenv("OPENARC_API_KEY")
model = OpenAIServerModel(
    model_id="Dolphin-X1",
    api_base="http://localhost:8000/v1",
    api_key=api_key,
    max_tokens=4096
)

# Add a system message to enforce JSON formatting
agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()], 
    model=model
)

answer = agent.run("Who discovered Pi")
print(answer)