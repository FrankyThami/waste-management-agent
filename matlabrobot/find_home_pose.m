function [qHome, ok, reason] = find_home_pose(robot, safety)
% find_home_pose.m
% Finds joint angles that put the FINGERTIPS at safety.homeXY, at height
% safety.homeTcpHeightM, with the gripper pointing straight down.
% The pose must pass the joint-limit check (with margin) and the floor check.
% Note: check_joint_limits takes (q, robot, safety); check_floor takes (robot, q, safety).

ok = false;
reason = "";
qHome = [];
eeName = "EndEffector_Link";

% Gripper pointing straight down: the tool's z-axis points along world -z.
R = [1 0 0; 0 -1 0; 0 0 -1];
tcp    = [safety.homeXY(:); safety.homeTcpHeightM];        % wanted fingertip position
flange = tcp - R * [0; 0; safety.tcpOffsetM];              % where the end of the arm must be
T = [R, flange; 0 0 0 1];

% Starting guesses for the solver: the old ready pose plus a few random poses.
savedRng = rng;
rng(0);                                                    % same random numbers on every run
qReady = deg2rad([0, 15, 180, -130, 0, 55, 90]);
nRandom = 6;
seeds = zeros(nRandom + 1, 7);
seeds(1, :) = qReady;
for k = 1:nRandom
    seeds(k + 1, :) = robot.randomConfiguration;
end
rng(savedRng);

ik = inverseKinematics("RigidBodyTree", robot);
ik.SolverParameters.AllowRandomRestart = false;
ik.SolverParameters.MaxTime = 0.3;

bestDist = inf;
lastReason = "no joint solution found for the home position";
for s = 1:size(seeds, 1)
    [q, info] = ik(eeName, T, [1 1 1 1 1 1], seeds(s, :));
    if ~strcmp(info.Status, "success")
        continue
    end
    [okJ, reasonJ] = check_joint_limits(q, robot, safety);
    if ~okJ
        lastReason = string(reasonJ);
        continue
    end
    [okF, reasonF] = check_floor(robot, q, safety);
    if ~okF
        lastReason = string(reasonF);
        continue
    end
    dist = norm(q - qReady);              % prefer the solution closest to the old ready pose
    if dist < bestDist
        bestDist = dist;
        qHome = q;
    end
end

ok = isfinite(bestDist);
if ok
    reason = "OK";
else
    reason = lastReason;
end

end