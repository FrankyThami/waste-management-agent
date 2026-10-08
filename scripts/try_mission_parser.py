"""Manual check: type a command and see what the real Gemini makes of it.
(This calls the API, so it uses a little quota.)"""

from src.reasoning.mission_parser import MissionParseError, parse_command

print("Type a sorting command. Press Enter on an empty line to quit.")
while True:
    command = input("\nCommand> ").strip()
    if not command:
        break
    try:
        print(parse_command(command).describe())
    except MissionParseError as error:
        print(f"Could not understand: {error}")