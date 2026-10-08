"""
agent/policy.py

Every tunable rule of the agent in ONE place: retry limits, safety
thresholds, loop ceilings. Guardrails, recovery and the run loop all
read from here, so changing a rule never means hunting through files.

Values marked PLACEHOLDER or ASSUMPTION must be confirmed once the
Kinova Gen3 and its gripper are available.
"""

# --- Recovery ---
MAX_PICK_ATTEMPTS = 2       # per physical object; after this the object is skipped
MAX_PLACE_ATTEMPTS = 2      # per held object; after this the task aborts (never carry on while holding something)
MATCH_TOLERANCE_M = 0.05    # positions closer than this (metres) are the SAME physical object

# --- Guardrails ---
MAX_OBJECT_WIDTH_M = 0.10                   # PLACEHOLDER - widest object the gripper can safely hold, in metres
APPROVAL_REQUIRED_CLASSES = {"unsorted"}    # ASSUMPTION - classes that ALWAYS pause for human approval
BORDERLINE_SIZE_FRACTION = 0.8              # ASSUMPTION - objects at/above 80% of the gripper limit also pause

# --- Agent loop ---
RECURSION_LIMIT = 60        # hard ceiling on agent steps per session

# --- Logging ---
LOG_DIR = "logs"

# --- Simulation (testing only - the real arm ignores all of these) ---
SIM_FORCED_GRASP_FAILURES = 0     # fail the next N grasps, then behave normally
SIM_FORCED_RELEASE_FAILURES = 0   # fail the next N releases, then behave normally
SIM_GRASP_FAILURE_RATE = 0.0      # chance (0.0 to 1.0) that any grasp fails at random
SIM_RELEASE_FAILURE_RATE = 0.0    # chance (0.0 to 1.0) that any release fails at random


# Fingertip height above a bin's rim when releasing an item (metres).
# EXAMPLE value: a safety margin, because the bin rims read about 0.5 cm low and a paper rim is thin.
# Tune it in the simulation.
RELEASE_CLEARANCE_M = 0.04

MAX_CONSECUTIVE_GUARD_REFUSALS = 3   # safety re-checks in a row that may refuse before the task is aborted