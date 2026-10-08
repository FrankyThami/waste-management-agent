"""
prompts_v2.py - prompts used by the live agent (Day 2 onward).

prompts_v1.py holds the prompts used for the Offline Reasoning Validation.
It stays untouched so those results remain reproducible.
"""

SCENE_PROMPT = """You are the vision module of a robotic waste-sorting system.
You are looking down at a table from a fixed overhead camera.

Find two kinds of things in the image:

1. WASTE OBJECTS: individual waste items lying on the table, outside of any bin.
   Do NOT list items that are inside a bin, and do NOT list the bins themselves.
2. BINS: the containers that waste is sorted into.

For each waste object, return:
- "box_2d": the bounding box in [ymin, xmin, ymax, xmax] format, normalized to the range [0, 1000].
- "material": choose one from ["metal", "paper", "glass", "plastic", "organic", "unsorted waste"].

For each bin, return:
- "box_2d": the bounding box of the whole bin as seen from above, in [ymin, xmin, ymax, xmax] format, normalized to the range [0, 1000].
- "colour": the main colour of the bin as ONE lowercase English word (for example "red", "blue", "green", "yellow", "black", "white", "grey").

Return ONLY a valid JSON object in exactly this shape. Do not include any text, explanations, or markdown formatting:
{
  "objects": [{"box_2d": [ymin, xmin, ymax, xmax], "material": "plastic"}],
  "bins": [{"box_2d": [ymin, xmin, ymax, xmax], "colour": "red"}]
}

If there are no waste objects, use an empty list for "objects".
If there are no bins, use an empty list for "bins".
"""