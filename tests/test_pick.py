from src.actions.tools import attempt_pick

reachable_target = {
    "id": "test_1",
    "waste_class": "plastic",
    "position": {"x": 0.1, "y": 0.1, "z": 0.1},
    "confidence": 1.0,
    "width_m": 0.05,
    "height_m": 0.05,
}

unreachable_target = {
    "id": "test_2",
    "waste_class": "metal",
    "position": {"x": 5.0, "y": 5.0, "z": 5.0},
    "confidence": 1.0,
    "width_m": 0.05,
    "height_m": 0.05,
}

print("Reachable target result:", attempt_pick(reachable_target))
print("Unreachable target result:", attempt_pick(unreachable_target))