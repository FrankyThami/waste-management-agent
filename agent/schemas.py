"""
Shared data shapes ("schemas") used by all 4 tools.

Defining these once means every tool speaks the same language - no
guessing field names, no silent typos between functions.
"""

from typing import TypedDict, List, Literal

# The 6 waste classes used throughout the project.
WasteClass = Literal["paper", "metal", "organic", "unsorted", "plastic", "glass"]


class Position(TypedDict):
    """A point in the robot base frame, in metres."""
    x: float
    y: float
    z: float


class DetectedObject(TypedDict):
    """One object detected by observe_scene()."""
    id: str
    waste_class: WasteClass
    position: Position
    confidence: float


# The result shape shared by attempt_pick, attempt_place, and safe_stop.
ToolStatus = Literal["success", "infeasible", "unreachable", "failed", "stopped"]


class ToolResult(TypedDict):
    status: ToolStatus
    reason: str