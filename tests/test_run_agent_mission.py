from unittest import mock

import src.interface.run_agent as run_agent
from src.context import state
from src.context.mission import BinRule, Mission
from src.reasoning.prompts import AGENT_SYSTEM_PROMPT, build_system_prompt


def make_mission():
    return Mission(
        command="plastic -> red bin, organic -> green bin; plastic first",
        bin_rules=[BinRule(waste_class="plastic", bin_color="red"), BinRule(waste_class="organic", bin_color="green")],
        sort_order=["plastic", "organic"],
    )


class FakeAgent:
    """Pretends to be the LangChain agent. Records what it was asked and
    which mission was active in state.py at that moment."""

    def __init__(self):
        self.payloads = []
        self.mission_during_run = "never ran"

    def invoke(self, payload, config=None):
        self.payloads.append(payload)
        self.mission_during_run = state.get_mission()
        return {"messages": []}


class FailingAgent:
    def invoke(self, payload, config=None):
        raise RuntimeError("boom")


class FakeLogger:
    """Replaces task_logger so these tests write nothing to your log files."""

    def __init__(self):
        self.events = []

    def start_session(self, instruction):
        self.events.append(("start", instruction))

    def end_session(self, reason, detail):
        self.events.append(("end", reason))


def run_with(fake_agent, mission=None):
    """Runs run_one_session() with the agent, the agent builder and the logger all replaced."""
    logger, built = FakeLogger(), []

    def fake_build(m=None):
        built.append(m)
        return fake_agent

    try:
        with mock.patch.object(run_agent, "build_agent", fake_build), \
             mock.patch.object(run_agent, "agent", fake_agent), \
             mock.patch.object(run_agent, "task_logger", logger):
            run_agent.run_one_session(mission=mission)
    finally:
        state.set_mission(None)
    return logger, built


def test_mission_run_builds_a_mission_agent_and_activates_the_mission():
    mission, fake = make_mission(), FakeAgent()
    logger, built = run_with(fake, mission)
    assert built == [mission]  # an agent was built FOR this mission
    assert fake.mission_during_run is mission  # state.py knew the mission while the agent ran
    assert mission.command in fake.payloads[0]["messages"][0]["content"]
    assert [event[0] for event in logger.events] == ["start", "end"]
    assert logger.events[1][1] == "finished"


def test_run_without_a_mission_behaves_exactly_as_before():
    fake = FakeAgent()
    logger, built = run_with(fake, None)
    assert built == []  # no new agent was built
    assert fake.mission_during_run is None
    assert fake.payloads[0]["messages"][0]["content"] == run_agent.DEFAULT_INSTRUCTION


def test_a_mission_never_carries_over_into_the_next_run():
    first, second = FakeAgent(), FakeAgent()
    run_with(first, make_mission())
    assert first.mission_during_run is not None
    run_with(second, None)
    assert second.mission_during_run is None


def test_an_error_is_logged_and_still_raised():
    logger = FakeLogger()
    raised = False
    try:
        with mock.patch.object(run_agent, "build_agent", lambda m=None: FailingAgent()), \
             mock.patch.object(run_agent, "task_logger", logger):
            run_agent.run_one_session(mission=make_mission())
    except RuntimeError:
        raised = True
    finally:
        state.set_mission(None)
    assert raised
    assert logger.events[-1] == ("end", "error")


def test_build_agent_passes_the_right_system_prompt():
    mission = make_mission()
    with mock.patch.object(run_agent, "create_agent") as fake_create:
        run_agent.build_agent(mission)
        run_agent.build_agent()
    with_mission, without_mission = fake_create.call_args_list
    assert with_mission.kwargs["system_prompt"] == build_system_prompt(mission)
    assert without_mission.kwargs["system_prompt"] == AGENT_SYSTEM_PROMPT


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