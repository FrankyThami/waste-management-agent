from agent import state

fake_objects = [
    {"id": "obj_0", "waste_class": "plastic", "position": {"x": 0, "y": 0, "z": 0}, "confidence": 1.0},
    {"id": "obj_1", "waste_class": "metal", "position": {"x": 0, "y": 0, "z": 0}, "confidence": 1.0},
]

state.reset(fake_objects)
print("After reset:", state.summary())

state.record_attempt("obj_0")
state.set_holding("obj_0")
state.mark_holding_done()
print("After finishing obj_0:", state.summary())

state.mark_holding_done()  # nothing held right now - should do nothing, not crash
print("Calling mark_holding_done with nothing held:", state.summary())