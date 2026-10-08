"""
Live accuracy check for the Mission Parser. THIS CALLS THE REAL GEMINI.

Why this exists: the unit tests in tests/ use a pretend Gemini, so they test our
checking code, not Gemini's understanding. This script feeds the real Gemini
sentences it has never been hinted about (new typos, new sentence shapes) and
compares the result with the answer a human would give.

Run from the project root:
    python -m scripts.eval_mission_parser
"""

import json
import os
import time

from src.reasoning.mission_parser import MissionParseError, parse_command

# "rules"  = class -> bin colour the mission must contain
# "order"  = the full sorting order the mission must end up with
# "refuse" = the parser must refuse (ask the user to rephrase) instead of guessing
CASES = [
    {"id": "c01_plain_list",
     "command": "Please sort everything: paper goes to the blue bin, metal to the grey bin, glass to the red bin.",
     "rules": {"paper": "blue", "metal": "grey", "glass": "red"}, "order": ["paper", "metal", "glass"]},
    {"id": "c02_explicit_order_first",
     "command": "Metal first, then glass, then paper. Metal -> silver bin, glass -> green bin, paper -> blue bin.",
     "rules": {"metal": "silver", "glass": "green", "paper": "blue"}, "order": ["metal", "glass", "paper"]},
    {"id": "c03_before_after_words",
     "command": "Before anything else handle the plastic, red bin. After that, organic waste into the brown bin.",
     "rules": {"plastic": "red", "organic": "brown"}, "order": ["plastic", "organic"]},
    {"id": "c04_typos_papre_metl",
     "command": "Put the papre in the blue bin and the metl in the black bin",
     "rules": {"paper": "blue", "metal": "black"}, "order": ["paper", "metal"]},
    {"id": "c05_two_classes_one_bin",
     "command": "Glass and metal both go in the grey bin.",
     "rules": {"glass": "grey", "metal": "grey"}, "order": ["glass", "metal"]},
    {"id": "c06_unsorted_waste_phrase",
     "command": "Organic waste into the purple bin, then the unsorted waste into the orange bin.",
     "rules": {"organic": "purple", "unsorted": "orange"}, "order": ["organic", "unsorted"]},
    {"id": "c07_only_one_class",
     "command": "Only sort the plastic, put it in the yellow bin.",
     "rules": {"plastic": "yellow"}, "order": ["plastic"]},
    {"id": "c08_glass_last",
     "command": "Do the glass last. Glass goes in the red bin, plastic goes in the blue bin and paper in the green bin.",
     "rules": {"glass": "red", "plastic": "blue", "paper": "green"}, "order": ["plastic", "paper", "glass"]},
    {"id": "c09_second_then_first",
     "command": "Second put the glass into the red bin, first put the paper into the blue bin.",
     "rules": {"glass": "red", "paper": "blue"}, "order": ["paper", "glass"]},
    {"id": "c10_colon_style_and_order_sentence",
     "command": "plastic: red. paper: blue. organic: green. Go in this order: organic, paper, plastic.",
     "rules": {"plastic": "red", "paper": "blue", "organic": "green"}, "order": ["organic", "paper", "plastic"]},
    {"id": "c11_capitals_typo_glsas",
     "command": "PUT THE GLSAS IN THE YELLOW BIN FIRST, THEN METAL IN THE GREY BIN",
     "rules": {"glass": "yellow", "metal": "grey"}, "order": ["glass", "metal"]},
    {"id": "c12_typos_orgnaic_plsatic",
     "command": "Orgnaic waste into the green bin and plsatic into the red bin.",
     "rules": {"organic": "green", "plastic": "red"}, "order": ["organic", "plastic"]},
    {"id": "c13_refuse_cans", "command": "put the cans into the grey bin", "refuse": True},
    {"id": "c14_refuse_no_waste_at_all", "command": "Clean the table and switch off the lights.", "refuse": True},
    {"id": "c15_refuse_contradiction", "command": "Plastic goes in the red bin and also in the blue bin.", "refuse": True},
    {"id": "c16_refuse_no_bin_given", "command": "Sort the plastic please.", "refuse": True},
]


def run_case(case):
    """Returns (status, detail). status is PASS, FAIL (wrong answer) or ERROR (API/network problem)."""
    try:
        mission = parse_command(case["command"])
    except MissionParseError as error:
        if case.get("refuse"):
            return "PASS", f"refused, as it should: {error}"
        return "FAIL", f"refused, but a human would understand this: {error}"
    except Exception as error:  # e.g. network or quota problems: not the parser's fault
        return "ERROR", f"{type(error).__name__}: {error}"

    got_rules = {rule.waste_class: rule.bin_color for rule in mission.bin_rules}
    got_order = list(mission.sort_order)
    if case.get("refuse"):
        return "FAIL", f"should have refused, but built rules={got_rules} order={got_order}"
    if got_rules == case["rules"] and got_order == case["order"]:
        return "PASS", ""
    return "FAIL", f"expected rules={case['rules']} order={case['order']} | got rules={got_rules} order={got_order}"


def main():
    results = []
    for case in CASES:
        status, detail = run_case(case)
        results.append({"id": case["id"], "command": case["command"], "status": status, "detail": detail})
        print(f"{status:5} {case['id']}")
        if status != "PASS":
            print(f"      {detail}")

    passed = sum(r["status"] == "PASS" for r in results)
    errors = sum(r["status"] == "ERROR" for r in results)
    print(f"\n{passed}/{len(results)} passed ({errors} infrastructure errors)")

    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", "mission_parser_eval.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, file, indent=2)
    print(f"Saved details to {path}")


if __name__ == "__main__":
    main()