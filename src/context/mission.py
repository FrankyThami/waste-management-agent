"""
A Mission is the clean, checked version of the sentence the user types.

Example sentence:
    "plastic -> red bin, paper -> blue bin, organic -> green bin;
     first plastic, then organics, finally paper"

becomes:
    bin_rules  = plastic->red, paper->blue, organic->green
    sort_order = plastic, organic, paper

The Mission also acts as a gatekeeper: if the parsed result is nonsense,
it is rejected here, before the robot ever moves.
"""

from typing import get_args

from pydantic import BaseModel, Field, field_validator, model_validator

from src.schemas import WasteClass

# The six waste classes, read from schemas.py so there is only ONE list.
VALID_WASTE_CLASSES: tuple[str, ...] = get_args(WasteClass)


class BinRule(BaseModel):
    """One rule: 'waste of this class goes into the bin of this colour'."""

    waste_class: WasteClass
    bin_color: str

    @field_validator("bin_color")
    @classmethod
    def clean_bin_color(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("bin_color cannot be empty")
        return value


class Mission(BaseModel):
    command: str  # the original sentence, kept for logs
    bin_rules: list[BinRule]
    sort_order: list[WasteClass] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_and_complete(self) -> "Mission":
        if not self.bin_rules:
            raise ValueError("A mission needs at least one bin rule")

        ruled_classes = [rule.waste_class for rule in self.bin_rules]

        if len(set(ruled_classes)) != len(ruled_classes):
            raise ValueError(
                "Each waste class can only have one bin rule (a class was given two bins)"
            )

        if len(set(self.sort_order)) != len(self.sort_order):
            raise ValueError("sort_order lists the same waste class more than once")

        for waste_class in self.sort_order:
            if waste_class not in ruled_classes:
                raise ValueError(
                    f"sort_order contains '{waste_class}' but no bin rule says where it goes"
                )

        # Any class that has a rule but was not named in the order goes last,
        # in the order it was mentioned. If no order was given, this fills it in.
        for waste_class in ruled_classes:
            if waste_class not in self.sort_order:
                self.sort_order.append(waste_class)

        return self

    def bin_color_for(self, waste_class: str) -> str | None:
        """Bin colour for a class, or None if the mission says nothing about it."""
        for rule in self.bin_rules:
            if rule.waste_class == waste_class:
                return rule.bin_color
        return None

    @property
    def untouched_classes(self) -> list[str]:
        """Classes the mission has no rule for. The agent leaves these alone."""
        return [c for c in VALID_WASTE_CLASSES if self.bin_color_for(c) is None]

    def describe(self) -> str:
        """A short human-readable summary (we will reuse this in Step 3)."""
        lines = ["Mission:"]
        for position, waste_class in enumerate(self.sort_order, start=1):
            lines.append(f"  {position}. {waste_class} -> {self.bin_color_for(waste_class)} bin")
        if self.untouched_classes:
            lines.append("  Will NOT touch: " + ", ".join(self.untouched_classes))
        return "\n".join(lines)