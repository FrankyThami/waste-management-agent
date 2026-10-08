"""
sim_layout.py - the simulated table, in ONE place.

All positions are FINGERTIP positions in metres, in the robot base frame
(x forward, y left, z up; z = 0 is the ground). EXAMPLE values for the
simulation only - replace them with real measurements for the real arm.
Bin release positions are NOT here: they live in config/matlab_bin_positions.json.
"""

# Must match safety_config.m (homeXY and homeTcpHeightM)
HOME = (0.30, 0.30, 0.050)

# Fingertip height when gripping an object that sits on the ground (metres).
PICK_Z = 0.025

# (waste class, x, y, z) - the pretend objects on the table
OBJECTS = [
    ("paper",   0.45,  0.00, PICK_Z),
    ("plastic", 0.45, -0.10, PICK_Z),
    ("organic", 0.45,  0.10, PICK_Z),
    ("metal",   0.38,  0.00, PICK_Z),
    ("glass",   0.90,  0.00, PICK_Z),   # out of reach on purpose
]

# Classes that are SUPPOSED to be unreachable (they test the 'unreachable' path).
UNREACHABLE_ON_PURPOSE = {"glass"}

# Positions used by the safety tests.
BELOW_FLOOR = (0.45, 0.00, 0.005)      # fingertips under the 10 mm floor limit
OUT_OF_REACH = (0.90, 0.00, 0.300)     # too far for the arm