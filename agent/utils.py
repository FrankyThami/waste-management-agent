"""
agent/utils.py

Small helpers shared across the agent's code.
"""


def get_response_text(message):
    """
    Extracts plain display text from a LangChain message's .content.

    Some models (like Gemini ER 2 with thinking enabled) return .content
    as a list of blocks instead of a plain string - one block is the
    actual text, others carry internal metadata (like reasoning
    signatures) that isn't meant to be displayed. This pulls out just
    the text, regardless of which shape .content comes back as.
    """
    if isinstance(message.content, str):
        return message.content

    parts = []
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
    return "".join(parts)