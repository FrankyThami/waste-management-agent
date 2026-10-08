"""
agent/tests/demo_recovery.py

Runs the REAL agent (real camera, real Gemini) but tells the simulated
arm to fail its first two grasps, so you can watch recovery happen live.

Run from the project root:
    python -m agent.tests.demo_recovery
"""

from src.actions import hal

hal.inject_failures(grasp=2)

from src.interface.run_agent import run_one_session, print_trace

result = run_one_session()
print_trace(result["messages"])