"""
scene_checks.py - sanity rules applied to a Scene before the agent trusts it (Stage 2.4).

The camera and Gemini can be wrong in predictable ways. These rules fix the obvious
cases and say so out loud - nothing is dropped silently:
  1. Duplicate detections: the same real item reported twice (same class or colour,
     boxes overlapping a lot) is merged into one.
  2. Items already in a bin: an object whose centre lies inside a bin's box is treated
     as already sorted and is not a target.
"""

from src.context.scene import Scene

DUPLICATE_OVERLAP = 0.6   # same kind + overlap this big (as a share of the SMALLER box) = same item


def _area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _overlap_of_smaller(a, b):
    """How much of the smaller box is covered by the other box (0.0 to 1.0)."""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller else 0.0


def _union(a, b):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _centre(box):
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


def _merge_duplicates(items, kind_of, id_of, notes):
    """Merges items of the same kind whose boxes overlap heavily. Returns the cleaned list."""
    kept_items = []
    for item in items:
        for kept in kept_items:
            if kind_of(kept) == kind_of(item) and _overlap_of_smaller(kept.box_px, item.box_px) >= DUPLICATE_OVERLAP:
                kept.box_px = _union(kept.box_px, item.box_px)
                kept.center_px = _centre(kept.box_px)
                notes.append(f"{id_of(item)} merged into {id_of(kept)} "
                             f"(same {kind_of(item)}, boxes overlap - one real item reported twice)")
                break
        else:
            kept_items.append(item)
    return kept_items


def _drop_items_in_bins(scene, notes):
    """Removes objects whose centre lies inside a bin's box."""
    targets = []
    for obj in scene.objects:
        cx, cy = obj.center_px
        holder = next((b for b in scene.bins
                       if b.box_px[0] <= cx <= b.box_px[2] and b.box_px[1] <= cy <= b.box_px[3]), None)
        if holder is None:
            targets.append(obj)
        else:
            notes.append(f"{obj.object_id} ({obj.waste_class}) is inside the {holder.colour} bin "
                         "- treated as already sorted, not a target")
    scene.objects = targets


def clean_scene(scene: Scene) -> list[str]:
    """Applies all rules to the Scene (changes it in place). Returns the notes, which are also printed."""
    notes = []
    scene.objects = _merge_duplicates(scene.objects, lambda o: o.waste_class, lambda o: o.object_id, notes)
    scene.bins = _merge_duplicates(scene.bins, lambda b: b.colour, lambda b: b.bin_id, notes)
    _drop_items_in_bins(scene, notes)
    for note in notes:
        print(f"[scene_checks] {note}")
    return notes