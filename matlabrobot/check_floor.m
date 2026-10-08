function [ok, reason] = check_floor(robot, q, safety)
% check_floor.m
% Floor-only safety rule: no checked point may be lower than safety.floorMinZ.
% Checked points:
%   1. the fingertips (end of the arm plus the gripper length along the tool's z-axis)
%   2. the origin of every arm link except the fixed base, which sits on the ground
% robot  = rigidBodyTree (DataFormat "row")
% q      = 1x7 joint angles in radians
% safety = settings from safety_config()

ok = true;
reason = 'OK';
floorZ = safety.floorMinZ;

% 1. Fingertips
Tee = getTransform(robot, q, "EndEffector_Link");
tcp = Tee(1:3, 4) + Tee(1:3, 1:3) * [0; 0; safety.tcpOffsetM];
if tcp(3) < floorZ
    ok = false;
    reason = sprintf('Fingertips at z = %.1f mm are below the %.0f mm floor limit', ...
                     1000 * tcp(3), 1000 * floorZ);
    return
end

% 2. Every arm link
for i = 1:robot.NumBodies
    bodyName = robot.BodyNames{i};
    T = getTransform(robot, q, bodyName);
    if T(3, 4) < floorZ
        ok = false;
        reason = sprintf('%s at z = %.1f mm is below the %.0f mm floor limit', ...
                         bodyName, 1000 * T(3, 4), 1000 * floorZ);
        return
    end
end

end