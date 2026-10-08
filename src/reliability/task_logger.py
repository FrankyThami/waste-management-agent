"""
agent/task_logger.py

The Automatic Task Logger. Two outputs, both written without the model
ever choosing to log anything:

  agent_log.csv       one row per tool call (raw evidence)
  session_summary.csv one row per run (the numbers your metrics come from)

The logger must NEVER break the agent. If a row can't be written (for
example the CSV is open in Excel, which locks it on Windows), it prints
a warning and the agent carries on.
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

from src import policy

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_FILE = _PROJECT_ROOT / policy.LOG_DIR / "agent_log.csv"
SUMMARY_FILE = _PROJECT_ROOT / policy.LOG_DIR / "session_summary.csv"

_LOG_COLUMNS = ["timestamp", "session_id", "event", "tool", "args", "status", "result", "duration_s"]

# Frozen before any real trials, so every row ever recorded has the same
# columns. Columns after the divider are RESERVED: they stay blank (never 0)
# until the step that measures them. Blank means "not measured".
_SUMMARY_COLUMNS = [
    "session_id", "timestamp", "outcome", "stop_reason", "completion_time_s",
    "tool_calls", "model_calls", "re_observations",
    "pick_attempts", "picks_succeeded", "place_attempts", "placements_succeeded",
    # --- reserved ---
    "retries", "scene_condition", "objects_in_scene", "correct_bin_placements", "task_success",
]

# Statuses our tools report as the first word of their result text
# (e.g. "success: Picked paper object ..."). Anything else is free text.
_KNOWN_STATUSES = {"success", "infeasible", "unreachable", "failed", "stopped", "rejected", "declined"}

# A pick/place only counts as an ATTEMPT if it actually reached the arm.
# Rejections (infeasible, unreachable, unknown id) are not attempts.
_ATTEMPT_STATUSES = {"success", "failed"}

# Session state - read and changed only through this module's functions.
_session_id = None
_started_at = None
_counts = {}
_last_summary = None


def _new_counts():
    return {
        "tool_calls": 0, "model_calls": 0, "observe_calls": 0,
        "pick_attempts": 0, "picks_succeeded": 0,
        "place_attempts": 0, "placements_succeeded": 0,
        "stop_reason": "",
    }


def _flatten(text):
    """One line per CSV row: newlines inside a result would make the file messy to read."""
    return " | ".join(str(text).splitlines())


def _parse_status(result_text):
    prefix = str(result_text).split(":", 1)[0].strip().lower()
    return prefix if prefix in _KNOWN_STATUSES else ""


def _append_csv(path, columns, fields):
    row = {col: "" for col in columns}
    row.update(fields)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError as e:
        print(f"[task_logger] WARNING: could not write to {path.name} ({e}). Is it open in Excel?")


def _write_log_row(**fields):
    fields.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    fields["session_id"] = _session_id or ""
    _append_csv(LOG_FILE, _LOG_COLUMNS, fields)


def _update_counts(tool, args, status):
    _counts["tool_calls"] += 1
    if tool == "observe_scene_tool":
        _counts["observe_calls"] += 1
    elif tool == "attempt_pick_tool" and status in _ATTEMPT_STATUSES:
        _counts["pick_attempts"] += 1
        if status == "success":
            _counts["picks_succeeded"] += 1
    elif tool == "attempt_place_tool" and status in _ATTEMPT_STATUSES:
        _counts["place_attempts"] += 1
        if status == "success":
            _counts["placements_succeeded"] += 1
    elif tool == "safe_stop_tool":
        _counts["stop_reason"] = str(args.get("reason", ""))


def start_session(instruction):
    """Call once at the start of a run. Returns the new session id."""
    global _session_id, _started_at, _counts
    _session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _started_at = time.perf_counter()
    _counts = _new_counts()
    _write_log_row(event="session_start", result=_flatten(instruction))
    return _session_id


def count_model_call():
    """Called once each time the agent's reasoning model is invoked."""
    _counts["model_calls"] += 1


def log_tool_call(tool, args, result, duration_s, status=None):
    """One row per tool call the agent makes, and one tick on the session counters."""
    status = status if status is not None else _parse_status(result)
    _update_counts(tool, args, status)
    _write_log_row(
        event="tool_call",
        tool=tool,
        args=json.dumps(args, default=str),
        status=status,
        result=_flatten(result),
        duration_s=f"{duration_s:.2f}",
    )


def end_session(outcome, detail=""):
    """Call once when a run ends. outcome is 'finished' or 'error'.
    Writes the session's summary row and remembers it for get_last_summary()."""
    global _last_summary
    elapsed = time.perf_counter() - _started_at if _started_at is not None else 0.0
    _write_log_row(event="session_end", status=outcome, result=_flatten(detail))

    summary = {
        "session_id": _session_id or "",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "outcome": outcome,
        "stop_reason": _flatten(detail) if outcome == "error" else _flatten(_counts["stop_reason"]),
        "completion_time_s": f"{elapsed:.1f}",
        "tool_calls": _counts["tool_calls"],
        "model_calls": _counts["model_calls"],
        "re_observations": max(_counts["observe_calls"] - 1, 0),
        "pick_attempts": _counts["pick_attempts"],
        "picks_succeeded": _counts["picks_succeeded"],
        "place_attempts": _counts["place_attempts"],
        "placements_succeeded": _counts["placements_succeeded"],
    }
    _last_summary = summary
    _append_csv(SUMMARY_FILE, _SUMMARY_COLUMNS, summary)


def get_last_summary():
    """The most recent session's summary dict, or None if no session has ended yet."""
    return _last_summary