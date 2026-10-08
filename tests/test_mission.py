from pydantic import ValidationError

from src.context.mission import Mission

COMMAND = (
    "sort the waste into the right bins in the live scene: plastic -> red bin, "
    "paper -> blue bin, organic -> green bin; first plastic, then organics, finally paper"
)


def rules(*pairs):
    """rules(("plastic", "red")) -> the list-of-dicts shape Gemini will give us later."""
    return [{"waste_class": c, "bin_color": b} for c, b in pairs]


def assert_rejected(build, why):
    """Passes only if building the Mission fails validation."""
    try:
        build()
    except ValidationError:
        return
    raise AssertionError(f"Expected the Mission to be rejected ({why}) but it was accepted")


def test_example_command_becomes_correct_mission():
    mission = Mission(
        command=COMMAND,
        bin_rules=rules(("plastic", "red"), ("paper", "blue"), ("organic", "green")),
        sort_order=["plastic", "organic", "paper"],
    )
    assert mission.sort_order == ["plastic", "organic", "paper"]
    assert mission.bin_color_for("organic") == "green"
    assert mission.bin_color_for("glass") is None
    assert mission.untouched_classes == ["metal", "unsorted", "glass"]


def test_order_defaults_to_order_mentioned():
    mission = Mission(command="x", bin_rules=rules(("plastic", "red"), ("paper", "blue")))
    assert mission.sort_order == ["plastic", "paper"]


def test_rule_missing_from_order_goes_last():
    mission = Mission(
        command="x",
        bin_rules=rules(("plastic", "red"), ("paper", "blue"), ("organic", "green")),
        sort_order=["paper"],
    )
    assert mission.sort_order == ["paper", "plastic", "organic"]


def test_bin_color_is_cleaned():
    mission = Mission(command="x", bin_rules=rules(("plastic", "  RED ")))
    assert mission.bin_color_for("plastic") == "red"


def test_unknown_waste_class_is_rejected():
    assert_rejected(
        lambda: Mission(command="x", bin_rules=rules(("cardboard", "red"))),
        "cardboard is not one of the six classes",
    )


def test_empty_mission_is_rejected():
    assert_rejected(lambda: Mission(command="x", bin_rules=[]), "no bin rules")


def test_same_class_twice_is_rejected():
    assert_rejected(
        lambda: Mission(command="x", bin_rules=rules(("plastic", "red"), ("plastic", "blue"))),
        "plastic was given two different bins",
    )


def test_order_with_class_that_has_no_bin_is_rejected():
    assert_rejected(
        lambda: Mission(
            command="x", bin_rules=rules(("plastic", "red")), sort_order=["plastic", "glass"]
        ),
        "glass is in the order but has no bin",
    )


def test_duplicate_in_order_is_rejected():
    assert_rejected(
        lambda: Mission(
            command="x",
            bin_rules=rules(("plastic", "red"), ("paper", "blue")),
            sort_order=["plastic", "plastic"],
        ),
        "plastic appears twice in the order",
    )


def test_empty_bin_color_is_rejected():
    assert_rejected(
        lambda: Mission(command="x", bin_rules=rules(("plastic", "   "))),
        "bin colour is blank",
    )


def test_describe_is_readable():
    mission = Mission(
        command=COMMAND,
        bin_rules=rules(("plastic", "red"), ("paper", "blue"), ("organic", "green")),
        sort_order=["plastic", "organic", "paper"],
    )
    text = mission.describe()
    assert "1. plastic -> red bin" in text
    assert "Will NOT touch: metal, unsorted, glass" in text


if __name__ == "__main__":
    demo = Mission(
        command=COMMAND,
        bin_rules=rules(("plastic", "red"), ("paper", "blue"), ("organic", "green")),
        sort_order=["plastic", "organic", "paper"],
    )
    print(demo.describe())
    print()

    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except Exception as error:
            failed += 1
            print(f"FAIL  {name}: {error}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")