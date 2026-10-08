% test_home_pose.m
% Finds the home pose, prints the facts about it, and draws it.

robot  = loadrobot("kinovaGen3", "DataFormat", "row");
safety = safety_config();

[qHome, ok, reason] = find_home_pose(robot, safety);
fprintf("\nHome pose found: ok=%d | %s\n", ok, reason);
if ~ok
    return
end

fprintf("Joint angles (deg): %s\n", mat2str(round(rad2deg(qHome), 1)));

Tee    = getTransform(robot, qHome, "EndEffector_Link");
flange = Tee(1:3, 4)';
tcp    = flange + (Tee(1:3, 1:3) * [0; 0; safety.tcpOffsetM])';
toolDir = Tee(1:3, 3)';

fprintf("Fingertips: x=%.3f  y=%.3f  z=%.3f m   (wanted z = %.3f)\n", tcp, safety.homeTcpHeightM);
fprintf("Gripper points along: [%.2f %.2f %.2f]   (straight down is [0 0 -1])\n", toolDir);
fprintf("Fingertip height above the ground: %.1f mm   (floor limit %.0f mm)\n", ...
        1000 * tcp(3), 1000 * safety.floorMinZ);

[okF, reasonF] = check_floor(robot, qHome, safety);
fprintf("Floor check:       ok=%d | %s\n", okF, reasonF);
[okJ, reasonJ] = check_joint_limits(qHome, robot, safety);
fprintf("Joint limit check: ok=%d | %s\n", okJ, reasonJ);

% --- Draw it ---
figure("Name", "Home pose");
ax = show(robot, qHome, "Frames", "off");
hold(ax, "on");

fz = safety.floorMinZ;     % red see-through plane = the "no-go below here" limit
patch(ax, [-0.6 1 1 -0.6], [-0.7 -0.7 0.7 0.7], [fz fz fz fz], "r", ...
      "FaceAlpha", 0.25, "EdgeColor", "none");
plot3(ax, [flange(1) tcp(1)], [flange(2) tcp(2)], [flange(3) tcp(3)], "g-", "LineWidth", 6);
plot3(ax, tcp(1), tcp(2), tcp(3), "ro", "MarkerSize", 8, "MarkerFaceColor", "r");

axis(ax, [-0.5 1 -0.7 0.7 0 1]);
view(ax, 135, 20);
title(ax, sprintf("Home pose: gripper down, fingertips %.0f mm above the ground", 1000 * tcp(3)));