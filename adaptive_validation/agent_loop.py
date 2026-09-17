"""
adaptive_validation/agent_loop.py

Shared tool-calling loop used by every agent script, so the
"ask model -> run whatever it requests -> feed result back" logic and
the response-parsing fix only exist in one place.
"""

from langchain_core.messages import ToolMessage


def extract_text(content) -> str:
    """Pull out readable text whether response.content is a plain
    string or a list of parts (this model sometimes attaches an
    internal reasoning signature alongside the text)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def run_tool_loop(llm, tool_map, messages, max_steps=16, pause_condition=None):
    """pause_condition(tool_name, tool_args) -> bool: checked after
    each tool executes. When it returns True, the script blocks on
    input() so you can physically change something before the model's
    next turn."""
    for step in range(max_steps):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\nFinal answer:\n{extract_text(response.content)}")
            return messages

        for call in response.tool_calls:
            print(f"\nStep {step + 1}: model calls {call['name']}({call['args']})")
            result = tool_map[call["name"]].invoke(call["args"])
            print(f"  -> {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

            if pause_condition and pause_condition(call["name"], call["args"]):
                input("\n>>> PAUSED - move the red dustbin now, then press Enter to continue... ")

    print("\nHit max_steps without a final answer - stopping.")
    return messages