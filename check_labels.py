"""Quick sanity check that labelled_scenes/labels.json is valid and loadable."""

import json

with open("labelled_scenes/labels.json") as f:
    data = json.load(f)

print(f"{len(data['scenes'])} scene(s) loaded")
print(f"First scene has {len(data['scenes'][0]['objects'])} objects")