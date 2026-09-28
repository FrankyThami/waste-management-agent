from agent.hal import check_reachable, simulate_move_to, simulate_grasp, simulate_release, get_bin_position

print("Reachable (0.1, 0.1, 0.1)?", check_reachable({"x": 0.1, "y": 0.1, "z": 0.1}))
print("Reachable (5.0, 5.0, 5.0)?", check_reachable({"x": 5.0, "y": 5.0, "z": 5.0}))

simulate_move_to({"x": 0.1, "y": 0.1, "z": 0.1})
simulate_grasp()
simulate_release()

print("Plastic bin position:", get_bin_position("plastic"))