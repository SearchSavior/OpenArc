

# I will leave this here for now but it wasnt working, something to do with imports?

from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=OpenAIServerModel(
    model_id="Qwen2.5-0.5B-Instruct-ov",
    api_base="http://localhost:8001/v1",
    api_key="openarc-api-key"))

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
