import json

from src.reasoning.mission_parser import _appears_in, build_prompt, parse_command
from tests.test_mission_parser import FakeGemini, expect_parse_error

# The real sentences that failed when typed by hand.
UNSORT_COMMAND = (
    "Look the scene and sort the wastes into right bins.Second put Organics wastes "
    "into the green bin and first put the unsort into yellow bin"
)
TYPO_COMMAND = (
    "Look the scene and sort the wastes into right bins. "
    "Organincs put to the yellow bin and plastic put into blue bin."
)


def test_abbreviation_and_reversed_order_are_accepted():
    reply = json.dumps({
        "bin_rules": [
            {"waste_class": "organic", "bin_color": "green"},
            {"waste_class": "unsorted", "bin_color": "yellow"},
        ],
        "sort_order": ["unsorted", "organic"],  # "first the unsort, second the organics"
        "unclear": [],
    })
    fake = FakeGemini(reply)
    mission = parse_command(UNSORT_COMMAND, ask=fake)
    assert mission.sort_order == ["unsorted", "organic"]
    assert mission.bin_color_for("unsorted") == "yellow"
    assert mission.bin_color_for("organic") == "green"
    assert len(fake.prompts) == 1  # accepted first time, no retry needed


def test_misspelling_is_accepted():
    reply = json.dumps({
        "bin_rules": [
            {"waste_class": "organic", "bin_color": "yellow"},
            {"waste_class": "plastic", "bin_color": "blue"},
        ],
        "sort_order": [],
        "unclear": [],
    })
    mission = parse_command(TYPO_COMMAND, ask=FakeGemini(reply))
    assert mission.sort_order == ["organic", "plastic"]
    assert mission.bin_color_for("organic") == "yellow"


def test_spelling_tolerance_has_limits():
    assert _appears_in("unsorted", "put the unsort into yellow bin")
    assert _appears_in("organic", "Organincs go in the bin")
    assert _appears_in("plastic", "plastics here")
    assert _appears_in("glass", "the glas bin")
    assert _appears_in("red", "reds")
    # a real word that merely LOOKS similar must NOT count:
    assert not _appears_in("unsorted", "sort the waste, everything sorted into bins")
    assert not _appears_in("orange", "plastic -> red, organic -> green")
    assert not _appears_in("glass", "put the class into the bin")
    assert not _appears_in("metal", "plastic -> red bin")


def test_invented_unsorted_rule_is_still_caught_when_command_says_sorted():
    # The command contains the word "sorted" but never mentions unsorted waste.
    command = "Once everything is sorted, put plastic in the red bin"
    invented = json.dumps({
        "bin_rules": [
            {"waste_class": "plastic", "bin_color": "red"},
            {"waste_class": "unsorted", "bin_color": "blue"},
        ],
        "sort_order": [], "unclear": [],
    })
    fixed = json.dumps({
        "bin_rules": [{"waste_class": "plastic", "bin_color": "red"}],
        "sort_order": [], "unclear": [],
    })
    fake = FakeGemini(invented, fixed)
    mission = parse_command(command, ask=fake)
    assert mission.bin_color_for("unsorted") is None
    assert len(fake.prompts) == 2  # the first answer was rejected


def test_different_word_is_still_unclear():
    reply = json.dumps({"bin_rules": [], "sort_order": [], "unclear": ["cans"]})
    message = expect_parse_error("put the cans in the grey bin", FakeGemini(reply))
    assert "cans" in message


def test_prompt_teaches_spelling_and_ordering():
    prompt = build_prompt(UNSORT_COMMAND)
    assert "unsort" in prompt
    assert "organincs" in prompt
    assert "second" in prompt.lower() and "first" in prompt.lower()


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