import os
from smolagents import CodeAgent, LiteLLMModel
from smolagents.default_tools import PythonInterpreterTool


def main():
    """Main entrypoint for interactive smolagent."""
    # Initialize the model using LiteLLM with OpenAI provider
    model = LiteLLMModel(
        model_id="openai/qwen25vl-3b",
        api_key=os.getenv("OPENARC_API_KEY"),
        api_base="http://localhost:8000/v1"
    )
    
    # Initialize only Python interpreter tool
    python_tool = PythonInterpreterTool()
    
    # Create the agent with only Python interpreter
    agent = CodeAgent(
        tools=[python_tool],
        model=model,
        max_steps=10
    )
    
    print("🤖 Smolagent with Python Interpreter")
    print("=" * 60)
    print("I can execute Python code!")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Run the agent
            print("\n🔄 Processing...\n")
            result = agent.run(user_input)
            print(f"\n🤖 Assistant: {result}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
