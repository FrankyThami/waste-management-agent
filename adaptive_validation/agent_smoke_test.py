"""
adaptive_validation/agent_smoke_test.py

Day 2, Step 1: prove LangChain can drive tool-calling with
gemini-robotics-er-2-preview specifically, before building any of the
real 8 tools on top of it. One trivial, unrelated tool only - if this
fails, the problem is the LangChain <-> model wiring, not our logic.

Run from the project root, venv active:
    python -m adaptive_validation.agent_smoke_test
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

load_dotenv()

MODEL_NAME = "gemini-robotics-er-2-preview"


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two whole numbers together and return the result."""
    print(f"    [tool executed] add_numbers({a}, {b})")
    return a + b


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found - check your .env file.")

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key)
    llm_with_tools = llm.bind_tools([add_numbers])

    prompt = "What is 47 plus 18? Use the add_numbers tool to work it out, don't do it in your head."
    print(f"Sending: {prompt!r}\n")
    response = llm_with_tools.invoke(prompt)

    if not response.tool_calls:
        print("No tool call came back - the model answered directly instead:")
        print(response.content)
        return
    for call in response.tool_calls:
        print(f"Model requested tool call: {call['name']}({call['args']})")
        if call["name"] == "add_numbers":
            result = add_numbers.invoke(call["args"])
            print(f"Tool result: {result}")


if __name__ == "__main__":
    main()