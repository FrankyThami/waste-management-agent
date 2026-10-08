% test_armsim_down.m
% Tests the upgraded ArmSim: home pose, downward gripper, floor-only safety.

fprintf("\nLoading ArmSim (finding the home pose)...\n");
arm = ArmSim();
fprintf("--- ArmSim: gripper down, floor-only safety ---\n");

s = arm.getState();
fprintf("1. Start (home): fingertips = [%.3f %.3f %.3f], gripper = %s, stopped = %d\n", ...
        s.tip, s.gripper, s.stopped);

show_result("2. check [0.50  0.00 0.050]            ", arm.checkPosition([0.50 0.00 0.050]));
show_result("3. move to [0.50  0.00 0.050] (anim)   ", arm.moveTo([0.50 0.00 0.050], true));
show_result("4. rise to [0.50  0.00 0.250] (anim)   ", arm.moveTo([0.50 0.00 0.250], true));
show_result("5. move to [0.40 -0.30 0.050] (anim)   ", arm.moveTo([0.40 -0.30 0.050], true));

s = arm.getState();
fprintf("   State now: fingertips = [%.3f %.3f %.3f]\n", s.tip);

show_result("6. below the floor [0.40 -0.30 0.005]  ", arm.moveTo([0.40 -0.30 0.005], false));
show_result("7. just above it   [0.40 -0.30 0.012]  ", arm.moveTo([0.40 -0.30 0.012], false));
show_result("8. out of reach    [0.90  0.00 0.300]  ", arm.moveTo([0.90 0.00 0.300], false));
show_result("9. close the gripper                   ", arm.setGripper("closed"));

arm.stop();
show_result("10. move after stop                    ", arm.moveTo([0.50 0.00 0.250], false));

show_result("11. reset                              ", arm.reset());
s = arm.getState();
fprintf("   State after reset: fingertips = [%.3f %.3f %.3f], gripper = %s, stopped = %d\n", ...
        s.tip, s.gripper, s.stopped);


function show_result(label, r)
fprintf("%s -> %s | %s | points=%d, time=%.2f s\n", label, r.status, r.reason, r.numPoints, r.durationS);
end