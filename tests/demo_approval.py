"""
agent/tests/demo_approval.py

Runs the REAL agent (real camera, real Gemini) but relabels the first
detected object as 'unsorted', so you can approve or deny it live
without needing a special item in view.

Run from the project root:
    python -m agent.tests.demo_approval
"""

from src.actions import langchain_tools as lt

_real_observe = lt.observe_scene

def _observe_with_one_flagged():
    objects = _real_observe()
    if objects:
        objects[0]["waste_class"] = "unsorted"
    return objects

lt.observe_scene = _observe_with_one_flagged

from src.interface.run_agent import run_one_session, print_trace

result = run_one_session()
print_trace(result["messages"])