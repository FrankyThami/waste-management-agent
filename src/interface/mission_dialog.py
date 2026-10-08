"""
Mission dialog: shows the user "Here's what I understood" and waits for a clear
YES before anything moves.

    typed sentence --> Mission Parser --> "Here's what I understood" --> y / n / q
                                                                          |
                                           y: return the Mission   n: retype   q: cancel

Safety rules built in:
  - There is NO default answer. Pressing Enter, or typing something unclear,
    never counts as "yes". After 3 unclear answers we cancel.
  - Ctrl+C or end-of-input cancels instead of crashing.
  - If Gemini is busy or unreachable, the user gets a friendly message and a
    chance to try again, not a Python traceback.

`ask_user`, `show` and `parse` can be swapped out, which is how the tests run
this whole conversation without a keyboard, a screen or Gemini.
"""

import logging
from typing import Callable

from src.context.mission import Mission
from src.reasoning.mission_parser import MissionParseError, parse_command

logger = logging.getLogger("mission_dialog")

YES_WORDS = {"y", "yes"}
NO_WORDS = {"n", "no"}
QUIT_WORDS = {"q", "quit"}
MAX_UNCLEAR_ANSWERS = 3

COMMAND_PROMPT = "What should the robot do? (type q to quit)\n> "
CONFIRM_PROMPT = "Is this right?   [y] yes, start   [n] no, let me retype   [q] quit\n> "
RETRY_PROMPT = "Try the same command again?   [y] yes   [n] no\n> "


def format_confirmation(mission: Mission) -> str:
    """The text shown to the user before anything moves."""
    return f'\nYou said: "{mission.command}"\n\nHere\'s what I understood:\n{mission.describe()}\n'


def _read(ask_user: Callable[[str], str], prompt: str) -> str | None:
    """Reads one answer. Returns None if the user pressed Ctrl+C or there is no more input."""
    try:
        return ask_user(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


def _ask_choice(ask_user, show, prompt: str, choices: dict[str, set[str]], hint: str) -> str | None:
    """Asks until the user gives a clear answer. `choices` maps a label (like "yes")
    to the words that mean it. Returns the label, or None if the user gave up or
    kept giving unclear answers."""
    for _ in range(MAX_UNCLEAR_ANSWERS):
        answer = _read(ask_user, prompt)
        if answer is None:
            return None
        answer = answer.strip().lower()
        for label, words in choices.items():
            if answer in words:
                return label
        show(f"Sorry, I need a clear answer: {hint}.")
    show("I didn't get a clear answer, so I'm cancelling to be safe.")
    return None


def get_confirmed_mission(
    first_command: str | None = None,
    ask_user: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    parse: Callable[[str], Mission] = parse_command,
) -> Mission | None:
    """
    Talks to the user until they confirm a mission or give up.
    Returns the confirmed Mission, or None if cancelled (the robot then does nothing).
    """
    command = first_command
    while True:
        # 1. Get a command to work with
        if command is None or not command.strip():
            command = _read(ask_user, COMMAND_PROMPT)
            if command is None or command.strip().lower() in QUIT_WORDS:
                show("Cancelled. Nothing was started.")
                return None
            continue

        # 2. Turn it into a Mission
        try:
            mission = parse(command)
        except MissionParseError as error:
            show(str(error))  # already written for the user, e.g. "I couldn't tell which class you meant by: bottles"
            command = None
            continue
        except Exception as error:  # normally Gemini busy / network problem
            logger.debug("Mission parsing failed", exc_info=True)
            detail = f"{type(error).__name__}: {error}"
            show(f"I couldn't get an answer from Gemini right now. It may be busy. ({detail[:150]})")
            retry = _ask_choice(ask_user, show, RETRY_PROMPT, {"yes": YES_WORDS, "no": NO_WORDS}, "y or n")
            if retry != "yes":
                show("Cancelled. Nothing was started.")
                return None
            continue  # same command, try again

        # 3. Show what we understood and wait for a clear yes
        show(format_confirmation(mission))
        choice = _ask_choice(
            ask_user, show, CONFIRM_PROMPT,
            {"yes": YES_WORDS, "no": NO_WORDS, "quit": QUIT_WORDS}, "y, n or q",
        )
        if choice == "yes":
            return mission
        if choice == "no":
            command = None
            continue
        show("Cancelled. Nothing was started.")
        return None