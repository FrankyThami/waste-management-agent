"""
Versioned prompt library — v1.

Adapted from the six benchmark reasoning prompts described in:

    Sinico, T.; Businaro, D.; Boschetti, G. Integrating Vision-Language-Action
    Models and RGB-D Sensing for Robotic Waste Sorting on KUKA LBR iiwa.
    Robotics 2026, 15, 100. https://doi.org/10.3390/robotics15050100
    (Open access, CC BY license)

The original paper found that prompt complexity trades off against inference
time: simpler prompts (1-3) are fastest but only identify objects one at a
time, while richer prompts (4-6) let the model reason about stacking order
and graspability directly, at the cost of ~1 extra second per call.

We reuse their six-class taxonomy directly, since it already matches our
project's categories.
"""

WASTE_CATEGORIES = ["metal", "paper", "glass", "plastic", "organic", "unsorted waste"]

# Shared role description used at the start of every prompt below.
AGENT_ROLE = (
    "You are an AI vision and grasp planning module for a robotic "
    "manipulation system."
)

# Shared closing instruction, to keep every response strictly parseable.
JSON_ONLY_INSTRUCTION = (
    "Return only the JSON list. Do not include any text, explanations, "
    "or markdown formatting."
)


PROMPT_1_OBJECT_CENTERS = f"""
{AGENT_ROLE} Identify all waste objects in the image.

For each object, return a JSON object with:
- "point": the center of the object in [y, x] format, normalized to 0-1000.
- "material": one of {WASTE_CATEGORIES}.

Example format:
[
  {{"point": [y1, x1], "material": "material1"}},
  {{"point": [y2, x2], "material": "material2"}}
]

{JSON_ONLY_INSTRUCTION}
"""

PROMPT_2_GRASP_POINTS = f"""
{AGENT_ROLE} Identify all waste objects in the image.

For each object, return a JSON object with:
- "point": the optimal grasp point in [y, x] format (0-1000), located on a
  stable, accessible part of the object's surface for a secure grasp.
- "material": one of {WASTE_CATEGORIES}.

{JSON_ONLY_INSTRUCTION}
"""

PROMPT_3_BOUNDING_BOXES = f"""
{AGENT_ROLE} Identify all waste objects in the image.

For each object, return a JSON object with:
- "box_2d": bounding box as [ymin, xmin, ymax, xmax], normalized to 0-1000.
- "material": one of {WASTE_CATEGORIES}.

{JSON_ONLY_INSTRUCTION}
"""

PROMPT_4_TOPMOST_OBJECT = f"""
{AGENT_ROLE} Perform a spatial analysis of the scene to identify the "depth
hierarchy" of the objects, then return only the single object physically
closest to the camera (i.e. least occluded, sitting on top of the stack).

Rules:
1. If object A overlaps the edge of object B, object A is higher.
2. The topmost object has zero occlusion from other items.
3. Ignore brightness; judge only by physical stacking.

Return a JSON list with exactly one entry:
- "point": geometric center [y, x] (0-1000).
- "material": one of {WASTE_CATEGORIES}.

{JSON_ONLY_INSTRUCTION}
"""

PROMPT_5_GRASP_PRIORITY = f"""
{AGENT_ROLE} Identify all objects and determine their stacking order.

For each object, return a JSON object with:
- "point": geometric center [y, x] (0-1000).
- "material": one of {WASTE_CATEGORIES}.
- "priority": integer grasp order. 1 = topmost/easiest to pick first;
  increasing values for objects further down the stack.

{JSON_ONLY_INSTRUCTION}
"""

PROMPT_6_GRASPABILITY = f"""
{AGENT_ROLE} Identify all waste objects and perform a strict occlusion
analysis to determine whether each can be safely picked up.

Rules:
1. An object is "graspable" only if it sits at the absolute top of the stack.
2. If any other object overlaps or rests on top of it, it is "not graspable".
3. If the edge of object A crosses over the surface of object B, B is
   "not graspable".
4. Visual size is irrelevant — partial coverage alone disqualifies it.

For each object, return a JSON object with:
- "point": geometric center [y, x] (0-1000).
- "material": one of {WASTE_CATEGORIES}.
- "status": "graspable" or "not graspable".

{JSON_ONLY_INSTRUCTION}
"""

# Convenient lookup so other scripts can grab a prompt by name.
ALL_PROMPTS = {
    "prompt_1_object_centers": PROMPT_1_OBJECT_CENTERS,
    "prompt_2_grasp_points": PROMPT_2_GRASP_POINTS,
    "prompt_3_bounding_boxes": PROMPT_3_BOUNDING_BOXES,
    "prompt_4_topmost_object": PROMPT_4_TOPMOST_OBJECT,
    "prompt_5_grasp_priority": PROMPT_5_GRASP_PRIORITY,
    "prompt_6_graspability": PROMPT_6_GRASPABILITY,
}