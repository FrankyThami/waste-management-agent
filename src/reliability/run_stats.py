"""
run_stats.py - counts the Gemini looks and safety re-checks of one run (Stage 3.4).

Every look at the table is one Gemini request, and the free tier allows only a few per day.
The agent's own thinking steps are counted separately in the session summary as "Model calls".
It also remembers how many safety re-checks IN A ROW refused to move the arm, so a scene that
never settles can be stopped (the limit lives in src/policy.py).
"""

_vision_looks = 0        # every observe() that asked Gemini
_guard_checks = 0        # safety re-checks before a pick or place
_guard_seconds = 0.0     # total time those re-checks took
_refusals_in_a_row = 0   # re-checks in a row that refused; any OK resets it


def count_vision_look():
    global _vision_looks
    _vision_looks += 1


def record_guard(ok, seconds):
    global _guard_checks, _guard_seconds, _refusals_in_a_row
    _guard_checks += 1
    _guard_seconds += seconds
    _refusals_in_a_row = 0 if ok else _refusals_in_a_row + 1


def consecutive_refusals():
    return _refusals_in_a_row


def report_and_reset():
    """One summary line for the run, then counting starts again from zero for the next run."""
    global _vision_looks, _guard_checks, _guard_seconds, _refusals_in_a_row
    line = (f"[stats] Gemini vision looks this run: {_vision_looks} "
            f"({_guard_checks} of them safety re-checks, {_guard_seconds:.0f} s in total). "
            "Add the 'Model calls' number from the session summary to get the total Gemini requests.")
    _vision_looks = 0
    _guard_checks = 0
    _guard_seconds = 0.0
    _refusals_in_a_row = 0
    return line