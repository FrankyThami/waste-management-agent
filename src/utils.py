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




# --- Waste class names ---------------------------------------------------
# Moved here from offline_reasoning/results_store.py so the agent does not
# depend on experiment code. Behaviour is unchanged.

# Canonical display labels, matching the reference table exactly.
# Handles the fact Gemini returns "unsorted waste" (from our prompt's
# category list) but the table header just says "Unsorted".
CLASS_LABELS = {
    "paper": "Paper",
    "metal": "Metal",
    "organic": "Organic",
    "unsorted waste": "Unsorted",
    "unsorted": "Unsorted",
    "plastic": "Plastic",
    "glass": "Glass",
}


def normalize_class(raw_class: str) -> str:
    """Turns "unsorted waste" / "PLASTIC" / etc. into the exact display
    label used in the table ("Unsorted" / "Plastic")."""
    key = raw_class.strip().lower()
    if key not in CLASS_LABELS:
        raise ValueError(
            f"Unknown waste class '{raw_class}'. Expected one of: paper, metal, organic, unsorted waste, plastic, glass"
        )
    return CLASS_LABELS[key]



