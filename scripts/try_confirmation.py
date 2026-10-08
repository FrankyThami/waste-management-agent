"""Manual check of the confirmation step with the REAL Gemini.
Run:  python -m scripts.try_confirmation"""

from src.interface.mission_dialog import get_confirmed_mission

mission = get_confirmed_mission()

if mission is None:
    print("\nNo mission confirmed. The robot would do nothing.")
else:
    print("\nConfirmed. This is exactly what the agent would receive:")
    print(mission.model_dump_json(indent=2))