"""
scene.py - the agent's structured picture of what the camera sees.

One Scene = one observation of the table. It answers three questions:
  1. What am I seeing?   -> objects (waste class) and bins (colour)
  2. Where are they?     -> pixel box + 3D position (camera frame and robot-base frame)
  3. What should I do?   -> answered later by matching a Mission against this Scene
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime

# A 3D point in metres: (x, y, z)
Position3D = tuple[float, float, float]


@dataclass
class SceneObject:
    """One waste item seen on the table."""
    object_id: str                              # e.g. "obj_1" - unique inside one Scene
    waste_class: str                            # e.g. "plastic"
    box_px: tuple[int, int, int, int]           # (xmin, ymin, xmax, ymax) in image pixels
    center_px: tuple[int, int]                  # (x, y) pixel where the robot should grasp
    position_camera: Position3D | None = None   # filled in Stage 2.3
    position_base: Position3D | None = None     # filled in Stage 2.3
    depth_valid: bool = False                   # True only if a real depth reading was found


@dataclass
class SceneBin:
    """One bin seen on the table."""
    bin_id: str                                 # e.g. "bin_1"
    colour: str                                 # e.g. "red"
    box_px: tuple[int, int, int, int]           # (xmin, ymin, xmax, ymax) in image pixels
    center_px: tuple[int, int]                  # (x, y) pixel of the bin centre
    position_camera: Position3D | None = None   # top-centre of the bin rim, Stage 2.3
    position_base: Position3D | None = None     # top-centre of the bin rim, Stage 2.3
    depth_valid: bool = False


@dataclass
class Scene:
    """Everything the agent sees in ONE observation."""
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    frame_path: str | None = None               # the saved camera image this came from
    objects: list[SceneObject] = field(default_factory=list)
    bins: list[SceneBin] = field(default_factory=list)

    def objects_of_class(self, waste_class: str) -> list[SceneObject]:
        """All objects of one class, e.g. every 'plastic' item."""
        return [o for o in self.objects if o.waste_class.lower() == waste_class.lower()]

    def find_bin(self, colour: str) -> SceneBin | None:
        """The bin of a given colour, or None if that colour is not visible."""
        for b in self.bins:
            if b.colour.lower() == colour.lower():
                return b
        return None

    def to_dict(self) -> dict:
        """Plain dictionary, used for saving the Scene to the log as JSON."""
        return asdict(self)

    def summary(self) -> str:
        """Human-readable text of the scene, for the terminal and for the agent."""
        lines = [
            f"Scene at {self.timestamp}: "
            f"{len(self.objects)} object(s), {len(self.bins)} bin(s)"
        ]
        for o in self.objects:
            lines.append(
                f"  {o.object_id}: {o.waste_class} at pixel {o.center_px}, "
                f"{_format_position(o.position_base)}"
            )
        for b in self.bins:
            lines.append(
                f"  {b.bin_id}: {b.colour} bin at pixel {b.center_px}, "
                f"{_format_position(b.position_base)}"
            )
        return "\n".join(lines)


def _format_position(position: Position3D | None) -> str:
    if position is None:
        return "position not measured yet"
    x, y, z = position
    return f"({x:.3f}, {y:.3f}, {z:.3f}) m"