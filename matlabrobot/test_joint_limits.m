% test_joint_limits.m
% Try some joint angle sets against the joint limits plus margin.

robot  = loadrobot("kinovaGen3", "DataFormat", "row");
safety = safety_config();

% Each row is 7 joint angles in DEGREES
testsDeg = [   0   15  180 -130   0   55   90;   % ready pose              -> OK
               0   15  180 -145   0   55   90;   % joint 4 inside margin   -> blocked
               0  130  180 -130   0   55   90;   % joint 2 past hard limit -> blocked
             720   15  180 -130   0   55   90;   % joint 1 spun round twice -> OK (no limit)
               0   15  180 -130   0  115   90 ]; % joint 6 inside margin   -> blocked

fprintf("\n--- Joint limit tests ---\n");
for i = 1:size(testsDeg, 1)
    q = deg2rad(testsDeg(i, :));
    [ok, reason] = check_joint_limits(q, robot, safety);
    fprintf("Test %d -> ok=%d | %s\n", i, ok, reason);
end