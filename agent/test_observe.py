from agent.tools import observe_scene

print("Point a waste item at the camera, then watch this window...\n")
objects = observe_scene()

print(f"\nDetected {len(objects)} object(s):")
for obj in objects:
    print(f"  {obj}")