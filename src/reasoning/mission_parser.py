"""
Mission Parser: turns the sentence the user types into a validated Mission.

    "plastic -> red bin, paper -> blue bin; first plastic, then paper"
         |
         v   (Gemini converts it to JSON)
    Mission(bin_rules=[...], sort_order=[...])

Safeguards:
  - temperature 0 (consistent answers)
  - Gemini fixes obvious spelling/abbreviations of the six class names
    ("organincs", "unsort"), but anything that is a genuinely different word
    ("bottles") goes into "unclear": we STOP and ask the user instead of guessing
  - every class and colour in the result must appear in the user's sentence,
    either exactly or as a close spelling of it (catches invented rules/colours)
  - if the answer is rejected, Gemini gets ONE more try that includes the reason
"""

import json
import re
from difflib import SequenceMatcher
from typing import Callable

from pydantic import ValidationError

from src.context.mission import VALID_WASTE_CLASSES, Mission

MAX_ATTEMPTS = 2
SPELLING_SIMILARITY = 0.8  # 1.0 = identical word, 0.0 = nothing in common


class MissionParseError(Exception):
    """The command could not be turned into a safe, valid Mission.
    The message is written so it can be shown directly to the user."""


# __CLASSES__ and __COMMAND__ are filled in by build_prompt() below.
_PROMPT_TEMPLATE = """You convert a robot operator's waste-sorting command into JSON.

The ONLY waste classes that exist are: __CLASSES__.

The operator types quickly, so spelling may be loose. Treat these as the class they obviously mean, and write the EXACT class name in your answer:
- "unsorted waste", "unsort", "unsorted" all mean the class unsorted
- plurals such as "organics", "plastics", "metals" mean organic, plastic, metal
- obvious misspellings such as "plasitc", "organincs", "glas" still mean that class
This is ONLY for spelling. If the word is a genuinely different thing (for example "bottles", "cans", "food", "cardboard"), do NOT map it to a class - put it in "unclear".

Return ONLY one JSON object, with no explanation and no markdown, in exactly this shape:
{
  "bin_rules": [{"waste_class": "<one of the classes>", "bin_color": "<colour word>"}],
  "sort_order": ["<class>", "<class>"],
  "unclear": ["<phrase>"]
}

Rules:
1. bin_rules: one entry per class that the command says where to put. Do NOT add rules for classes the command does not mention.
2. bin_color: just the colour word exactly as the user wrote it, in lowercase ("red", not "the red bin").
3. sort_order: use it when the command states an order with words like first, second, third, then, finally, before, after. The order in which classes are MENTIONED is not always the sorting order: "second put X ... and first put Y" means Y is sorted before X. If no order is stated, return an empty list [].
4. unclear: if the command names something that is not one of the classes (and is not just a spelling of one), or you cannot tell which class or colour is meant, put that phrase in "unclear". NEVER guess a class.

Example 1
Command: glass -> yellow bin, metal -> grey bin, do metal first
Output: {"bin_rules": [{"waste_class": "glass", "bin_color": "yellow"}, {"waste_class": "metal", "bin_color": "grey"}], "sort_order": ["metal", "glass"], "unclear": []}

Example 2
Command: put the bottles in the blue bin
Output: {"bin_rules": [], "sort_order": [], "unclear": ["bottles"]}

Example 3
Command: Look at the scene. Second, put the metals into the grey bin. First, put the glas into the red bin.
Output: {"bin_rules": [{"waste_class": "metal", "bin_color": "grey"}, {"waste_class": "glass", "bin_color": "red"}], "sort_order": ["glass", "metal"], "unclear": []}

Now convert this command:
__COMMAND__
"""


def build_prompt(command: str, previous_error: str | None = None) -> str:
    """Builds the text we send to Gemini. If a previous answer was rejected,
    the reason is added so Gemini can fix it."""
    prompt = _PROMPT_TEMPLATE.replace("__CLASSES__", ", ".join(VALID_WASTE_CLASSES))
    prompt = prompt.replace("__COMMAND__", command.strip())
    if previous_error:
        prompt += (
            f"\nYour previous answer was rejected: {previous_error}\n"
            "Fix the problem and return the corrected JSON only.\n"
        )
    return prompt


def _extract_json(text: str) -> dict:
    """Finds the JSON object inside Gemini's reply, even if it added ```json fences or chatter."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("the reply did not contain a JSON object")
    return json.loads(text[start : end + 1])  # a bad JSON error is also a ValueError


def _appears_in(word: str, command: str) -> bool:
    """True if `word` is in the command, either exactly or as a close spelling.

    Accepted: the exact word, a plural or small typo ("organincs" for organic),
    or a cut-short version of at least 4 letters ("unsort" for unsorted).
    The FIRST LETTER must match. This stops the ordinary word "sorted" from
    counting as the class "unsorted", even though the two look alike.
    """
    word = word.lower()
    for command_word in re.findall(r"[a-z]+", command.lower()):
        if command_word[0] != word[0]:
            continue
        if command_word == word:
            return True
        if len(command_word) >= 4 and word.startswith(command_word):
            return True
        if SequenceMatcher(None, command_word, word).ratio() >= SPELLING_SIMILARITY:
            return True
    return False


def _check_grounded(mission: Mission, command: str) -> None:
    """Every class and colour must appear in what the user typed."""
    for rule in mission.bin_rules:
        for word, label in ((rule.waste_class, "waste class"), (rule.bin_color, "bin colour")):
            if not _appears_in(word, command):
                raise ValueError(
                    f"the {label} '{word}' does not appear in the user's command; "
                    "only use words the user actually wrote"
                )


def _summarize_validation_error(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        where = ".".join(str(p) for p in item["loc"]) or "mission"
        parts.append(f"{where}: {item['msg']}")
    return "; ".join(parts)


def parse_command(command: str, ask: Callable[[str], str] | None = None) -> Mission:
    """
    Turns a typed command into a Mission, or raises MissionParseError.

    `ask` is the function that sends a prompt to Gemini and returns the reply
    text. Normally you leave it out (we use the real Gemini). Tests pass in a
    fake one so they need no internet.
    """
    if not command or not command.strip():
        raise MissionParseError("The command is empty. Please type what you want the robot to sort.")

    if ask is None:
        from src.models.gemini_classifier import generate_text  # imported here so tests don't need an API key

        ask = generate_text

    previous_error = None
    for _attempt in range(MAX_ATTEMPTS):
        reply = ask(build_prompt(command, previous_error))
        try:
            data = _extract_json(reply)

            unclear = [str(item) for item in (data.get("unclear") or [])]
            if unclear:
                raise MissionParseError(
                    f"I couldn't tell which waste class you meant by: {', '.join(unclear)}. "
                    f"The classes I know are: {', '.join(VALID_WASTE_CLASSES)}. Please rephrase."
                )

            mission = Mission(
                command=command.strip(),
                bin_rules=data.get("bin_rules", []),
                sort_order=data.get("sort_order", []),
            )
            _check_grounded(mission, command)
            return mission

        except ValidationError as error:  # must come before ValueError (it is a kind of ValueError)
            previous_error = _summarize_validation_error(error)
        except ValueError as error:
            previous_error = str(error)

    raise MissionParseError(
        f"I couldn't turn that into a valid mission (last problem: {previous_error}). "
        f"Please rephrase using the class names ({', '.join(VALID_WASTE_CLASSES)}) "
        "and name each bin by its colour."
    )