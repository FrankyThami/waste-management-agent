from src.context import state
from src.context.mission import BinRule, Mission
from src.reasoning.prompts import AGENT_SYSTEM_PROMPT, build_system_prompt


def pos(x):
    return {"x": x, "y": 0.0, "z": 0.0}  # positions 0.3 m apart are different physical objects


OBJECTS = [
    {"id": "obj_0", "waste_class": "plastic", "position": pos(0.0)},
    {"id": "obj_1", "waste_class": "organic", "position": pos(0.3)},
    {"id": "obj_2", "waste_class": "paper", "position": pos(0.6)},
    {"id": "obj_3", "waste_class": "metal", "position": pos(0.9)},
    {"id": "obj_4", "waste_class": "plastic", "position": pos(1.2)},
]


def make_mission():
    """plastic first, then organic. Paper, metal, unsorted and glass are not part of it."""
    return Mission(
        command="plastic -> red bin, organic -> green bin; plastic first",
        bin_rules=[BinRule(waste_class="plastic", bin_color="red"), BinRule(waste_class="organic", bin_color="green")],
        sort_order=["plastic", "organic"],
    )


def with_clean_state(test):
    """Starts and ends every test with an empty state and no mission."""
    def wrapper():
        state.new_session()
        state.set_mission(None)
        try:
            test()
        finally:
            state.new_session()
            state.set_mission(None)

    wrapper.__name__ = test.__name__
    return wrapper


@with_clean_state
def test_without_a_mission_nothing_changes():
    state.reset(OBJECTS)
    for obj in OBJECTS:
        assert state.check_can_pick(obj["id"]) is None
    assert len(state.pending_ids()) == 5
    assert "Left alone" not in state.summary()


@with_clean_state
def test_class_outside_the_mission_is_rejected_and_not_skipped():
    state.reset(OBJECTS)
    state.set_mission(make_mission())
    message = state.check_can_pick("obj_2")  # paper
    assert message.startswith("rejected")
    assert "does not include 'paper'" in message
    assert state._objects["obj_2"]["status"] == "pending"  # left alone, not marked as skipped or failed


@with_clean_state
def test_order_is_enforced_in_code():
    state.reset(OBJECTS)
    state.set_mission(make_mission())
    message = state.check_can_pick("obj_1")  # organic, but plastic is still pending
    assert message.startswith("rejected")
    assert "'plastic' before 'organic'" in message
    assert "obj_0" in message and "obj_4" in message
    assert state.check_can_pick("obj_0") is None  # plastic itself is allowed
    assert state.check_can_pick("obj_4") is None


@with_clean_state
def test_next_class_is_allowed_once_the_earlier_class_is_finished_or_skipped():
    state.reset(OBJECTS)
    state.set_mission(make_mission())
    state.set_holding("obj_0")
    state.mark_holding_done()  # obj_0 sorted
    assert state.check_can_pick("obj_1") is not None  # obj_4 (plastic) still pending
    state.skip("obj_4")  # the agent gave up on it
    assert state.check_can_pick("obj_1") is None  # now organic may start


@with_clean_state
def test_pending_and_summary_ignore_objects_outside_the_mission():
    state.reset(OBJECTS)
    state.set_mission(make_mission())
    assert state.pending_ids() == ["obj_0", "obj_1", "obj_4"]
    summary = state.summary()
    assert "3 object(s) still pending: obj_0 (plastic), obj_1 (organic), obj_4 (plastic)" in summary
    assert "Left alone" in summary and "obj_2 (paper)" in summary and "obj_3 (metal)" in summary


@with_clean_state
def test_only_out_of_mission_objects_means_nothing_pending():
    state.reset([OBJECTS[2], OBJECTS[3]])  # paper and metal only
    state.set_mission(make_mission())
    assert state.pending_ids() == []
    assert state.summary().startswith("No pending objects remain")


@with_clean_state
def test_clearing_the_mission_restores_normal_behaviour():
    state.reset(OBJECTS)
    state.set_mission(make_mission())
    assert state.check_can_pick("obj_1") is not None
    state.set_mission(None)
    assert state.get_mission() is None
    assert state.check_can_pick("obj_1") is None
    assert len(state.pending_ids()) == 5


@with_clean_state
def test_existing_rules_still_win_over_the_mission():
    # holding something must still block every pick, mission or not
    state.reset(OBJECTS)
    state.set_mission(make_mission())
    state.set_holding("obj_0")
    assert "still holding" in state.check_can_pick("obj_4")
    # an aborted task must still block every pick
    state.set_holding("obj_0")
    state.mark_holding_done()
    state.abort("test")
    assert "aborted" in state.check_can_pick("obj_4")


def test_prompt_without_mission_is_the_original_prompt():
    assert build_system_prompt(None) == AGENT_SYSTEM_PROMPT
    assert build_system_prompt() == AGENT_SYSTEM_PROMPT


def test_prompt_with_mission_states_the_plan_and_keeps_the_original():
    prompt = build_system_prompt(make_mission())
    assert prompt.startswith(AGENT_SYSTEM_PROMPT)
    assert 'The operator said: "plastic -> red bin, organic -> green bin; plastic first"' in prompt
    assert "1. plastic -> red bin" in prompt
    assert "2. organic -> green bin" in prompt
    assert "paper, metal, unsorted, glass" in prompt  # the classes to leave alone
    assert "safe_stop_tool" in prompt
    assert "do not claim an object went into a coloured bin" in prompt


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