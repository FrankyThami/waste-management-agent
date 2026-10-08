import json

from src.context.mission import VALID_WASTE_CLASSES
from src.reasoning.mission_parser import MissionParseError, build_prompt, parse_command

COMMAND = (
    "sort the waste into the right bins in the live scene: plastic -> red bin, "
    "paper -> blue bin, organic -> green bin; first plastic, then organics, finally paper"
)

GOOD = {
    "bin_rules": [
        {"waste_class": "plastic", "bin_color": "red"},
        {"waste_class": "paper", "bin_color": "blue"},
        {"waste_class": "organic", "bin_color": "green"},
    ],
    "sort_order": ["plastic", "organic", "paper"],
    "unclear": [],
}


class FakeGemini:
    """Pretends to be Gemini. Returns the replies we give it, one per call,
    and remembers every prompt it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def expect_parse_error(command, fake):
    try:
        parse_command(command, ask=fake)
    except MissionParseError as error:
        return str(error)
    raise AssertionError("Expected MissionParseError but the command was accepted")


def test_good_reply_gives_mission():
    fake = FakeGemini(json.dumps(GOOD))
    mission = parse_command(COMMAND, ask=fake)
    assert mission.sort_order == ["plastic", "organic", "paper"]
    assert mission.bin_color_for("paper") == "blue"
    assert mission.command == COMMAND
    assert len(fake.prompts) == 1


def test_fenced_reply_with_chatter_is_accepted():
    reply = "Sure! Here is the JSON:\n```json\n" + json.dumps(GOOD) + "\n```"
    mission = parse_command(COMMAND, ask=FakeGemini(reply))
    assert mission.bin_color_for("organic") == "green"


def test_invalid_class_triggers_one_retry_with_feedback():
    bad = {"bin_rules": [{"waste_class": "cardboard", "bin_color": "red"}], "sort_order": [], "unclear": []}
    fake = FakeGemini(json.dumps(bad), json.dumps(GOOD))
    mission = parse_command(COMMAND, ask=fake)
    assert mission.bin_color_for("plastic") == "red"
    assert len(fake.prompts) == 2
    assert "rejected" not in fake.prompts[0]
    assert "rejected" in fake.prompts[1]


def test_invented_colour_is_caught():
    bad = {**GOOD, "bin_rules": [
        {"waste_class": "plastic", "bin_color": "orange"},
        {"waste_class": "paper", "bin_color": "blue"},
        {"waste_class": "organic", "bin_color": "green"},
    ]}
    fake = FakeGemini(json.dumps(bad), json.dumps(GOOD))
    mission = parse_command(COMMAND, ask=fake)
    assert mission.bin_color_for("plastic") == "red"
    assert "orange" in fake.prompts[1]


def test_invented_rule_for_unmentioned_class_is_caught():
    bad = {**GOOD, "bin_rules": GOOD["bin_rules"] + [{"waste_class": "glass", "bin_color": "yellow"}]}
    fake = FakeGemini(json.dumps(bad), json.dumps(GOOD))
    mission = parse_command(COMMAND, ask=fake)
    assert mission.bin_color_for("glass") is None
    assert len(fake.prompts) == 2


def test_unparseable_reply_retries_then_succeeds():
    fake = FakeGemini("I'm sorry, I cannot do that", json.dumps(GOOD))
    mission = parse_command(COMMAND, ask=fake)
    assert mission.sort_order == ["plastic", "organic", "paper"]
    assert len(fake.prompts) == 2


def test_gives_up_after_two_bad_replies():
    fake = FakeGemini("nonsense", "still nonsense")
    expect_parse_error(COMMAND, fake)
    assert len(fake.prompts) == 2


def test_unclear_phrase_stops_without_retry():
    reply = json.dumps({"bin_rules": [], "sort_order": [], "unclear": ["bottles"]})
    fake = FakeGemini(reply)
    message = expect_parse_error("put the bottles in the blue bin", fake)
    assert "bottles" in message
    assert len(fake.prompts) == 1


def test_empty_command_never_calls_gemini():
    fake = FakeGemini()
    expect_parse_error("   ", fake)
    assert len(fake.prompts) == 0


def test_prompt_contains_classes_and_command():
    prompt = build_prompt(COMMAND)
    assert COMMAND in prompt
    for waste_class in VALID_WASTE_CLASSES:
        assert waste_class in prompt
    assert "rejected" not in prompt
    retry_prompt = build_prompt(COMMAND, "oops")
    assert "rejected" in retry_prompt and "oops" in retry_prompt


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