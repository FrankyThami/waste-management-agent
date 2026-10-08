from src.context.mission import BinRule, Mission
from src.interface.mission_dialog import format_confirmation, get_confirmed_mission
from src.reasoning.mission_parser import MissionParseError


def make_mission(command="plastic -> red bin, paper -> blue bin"):
    return Mission(
        command=command,
        bin_rules=[BinRule(waste_class="plastic", bin_color="red"), BinRule(waste_class="paper", bin_color="blue")],
    )


class Script:
    """Pretends to be a person typing. Gives the answers we list, one per question.
    If the program asks more questions than we prepared, it behaves like Ctrl+D (end of input)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


class FakeParser:
    """Pretends to be the Mission Parser. Each call returns (or raises) the next item we queued."""

    def __init__(self, *results):
        self.results = list(results)
        self.commands = []

    def __call__(self, command):
        self.commands.append(command)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def run(script, parser, first_command=None):
    shown = []
    mission = get_confirmed_mission(first_command, ask_user=script, show=shown.append, parse=parser)
    return mission, "\n".join(shown)


def test_yes_returns_the_mission_without_asking_for_the_command():
    mission = make_mission()
    script, parser = Script("y"), FakeParser(mission)
    result, shown = run(script, parser, first_command="my command")
    assert result is mission
    assert parser.commands == ["my command"]
    assert len(script.prompts) == 1  # only the confirmation question was asked


def test_confirmation_text_shows_the_sentence_and_the_mission():
    text = format_confirmation(make_mission("plastic -> red bin, paper -> blue bin"))
    assert 'You said: "plastic -> red bin, paper -> blue bin"' in text
    assert "Here's what I understood:" in text
    assert "1. plastic -> red bin" in text
    assert "Will NOT touch" in text


def test_asks_for_the_command_when_none_is_given():
    mission = make_mission()
    script, parser = Script("sort the plastic", "yes"), FakeParser(mission)
    result, _ = run(script, parser)
    assert result is mission
    assert parser.commands == ["sort the plastic"]


def test_no_lets_the_user_retype_and_uses_the_new_command():
    first, second = make_mission("first"), make_mission("second")
    script, parser = Script("n", "second try", "y"), FakeParser(first, second)
    result, _ = run(script, parser, first_command="first try")
    assert result is second
    assert parser.commands == ["first try", "second try"]


def test_quit_at_confirmation_cancels():
    script, parser = Script("q"), FakeParser(make_mission())
    result, shown = run(script, parser, first_command="x")
    assert result is None
    assert "Nothing was started" in shown


def test_quit_when_asked_for_a_command_cancels():
    script, parser = Script("quit"), FakeParser()
    result, _ = run(script, parser)
    assert result is None
    assert parser.commands == []


def test_unclear_answers_are_never_treated_as_yes():
    script, parser = Script("", "maybe", "ok sure"), FakeParser(make_mission())
    result, shown = run(script, parser, first_command="x")
    assert result is None
    assert "cancelling to be safe" in shown


def test_unclear_answer_then_clear_yes_is_accepted():
    mission = make_mission()
    script, parser = Script("maybe", "  YES  "), FakeParser(mission)
    result, shown = run(script, parser, first_command="x")
    assert result is mission
    assert "I need a clear answer" in shown


def test_parse_error_is_shown_and_user_can_retype():
    mission = make_mission()
    error = MissionParseError("I couldn't tell which waste class you meant by: bottles. Please rephrase.")
    script, parser = Script("plastic -> red bin", "y"), FakeParser(error, mission)
    result, shown = run(script, parser, first_command="put the bottles in the blue bin")
    assert result is mission
    assert "bottles" in shown
    assert parser.commands == ["put the bottles in the blue bin", "plastic -> red bin"]


def test_gemini_failure_gives_friendly_message_and_retries_same_command():
    mission = make_mission()
    busy = RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
    script, parser = Script("y", "y"), FakeParser(busy, mission)  # y = retry, y = confirm
    result, shown = run(script, parser, first_command="my command")
    assert result is mission
    assert parser.commands == ["my command", "my command"]  # SAME command, not a new one
    assert "may be busy" in shown
    assert "Traceback" not in shown


def test_gemini_failure_then_no_cancels():
    script, parser = Script("n"), FakeParser(RuntimeError("503"))
    result, shown = run(script, parser, first_command="x")
    assert result is None
    assert "Nothing was started" in shown


def test_end_of_input_cancels_instead_of_crashing():
    result, _ = run(Script(), FakeParser(make_mission()), first_command="x")
    assert result is None
    result, _ = run(Script(), FakeParser())  # no command, no answers
    assert result is None


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except Exception as error:
            failed += 1
            print(f"FAIL  {name}: {error!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")