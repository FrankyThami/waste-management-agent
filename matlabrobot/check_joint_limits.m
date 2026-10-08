function [ok, reason] = check_joint_limits(q, robot, safety)
% check_joint_limits.m
% q      = 1x7 row of joint angles in radians
% robot  = the rigidBodyTree from loadrobot (DataFormat "row")
% safety = the settings from safety_config()
% ok     = true if every joint is inside its limits minus the safety margin
% reason = a short message explaining the decision

margin = deg2rad(safety.jointMarginDeg);
ok = true;
reason = 'OK';

jointIndex = 0;
for i = 1:robot.NumBodies
    joint = robot.Bodies{i}.Joint;
    if strcmp(joint.Type, "fixed")
        continue            % skip the fixed joint at the end
    end
    jointIndex = jointIndex + 1;

    lims = joint.PositionLimits;
    if ~isfinite(lims(1)) || ~isfinite(lims(2))
        continue            % this joint can spin freely, nothing to check
    end

    safeMin = lims(1) + margin;
    safeMax = lims(2) - margin;

    if q(jointIndex) < safeMin || q(jointIndex) > safeMax
        ok = false;
        reason = sprintf('Joint %d (%s) at %.1f deg is outside the safe range [%.1f, %.1f] deg', ...
                         jointIndex, joint.Name, rad2deg(q(jointIndex)), ...
                         rad2deg(safeMin), rad2deg(safeMax));
        return
    end
end

end