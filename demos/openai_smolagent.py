from smolagents import ToolCallingAgent, DuckDuckGoSearchTool
from smolagents.models import OpenAIServerModel
import os


api_key = os.getenv("OPENARC_API_KEY")
model = OpenAIServerModel(
    model_id="Qwen3-4B-2507",
    api_base="http://localhost:8000/v1",
    api_key=api_key
)

agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=model)

answer = agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
print(answer)