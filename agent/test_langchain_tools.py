from agent.langchain_tools import (
    observe_scene_tool, attempt_pick_tool, attempt_place_tool,
    safe_stop_tool, _last_observation,
)

print("=== observe_scene_tool ===")
print(observe_scene_tool.invoke({}))

if _last_observation:
    first_id = next(iter(_last_observation))
    waste_class = _last_observation[first_id]["waste_class"]

    print(f"\n=== attempt_pick_tool ({first_id}) ===")
    print(attempt_pick_tool.invoke({"target_id": first_id}))

    print(f"\n=== attempt_place_tool ({waste_class}) ===")
    print(attempt_place_tool.invoke({"waste_class": waste_class}))
else:
    print("\nNo objects detected - point an item at the camera and re-run to test pick/place.")

print("\n=== safe_stop_tool ===")
print(safe_stop_tool.invoke({"reason": "End of Step 2 test"}))