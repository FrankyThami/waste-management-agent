% explore_down_pose.m
% Stage G1 - Where can the gripper point straight DOWN with its fingertips
% 40 mm above the ground? Tries a grid of positions and marks the reachable ones.

robot  = loadrobot("kinovaGen3", "DataFormat", "row");
safety = safety_config();          % used here only for the joint-limit margin
eeName = "EndEffector_Link";

% --- Settings you can change ---
tcpOffsetM = safety.tcpOffsetM;        % gripper length, from safety_config.m
tcpZ       = safety.homeTcpHeightM;    % wanted fingertip height, from safety_config.m
xs = 0.20:0.10:0.60;    % forward distance from the base (metres)
ys = -0.30:0.15:0.30;   % left (+) / right (-) of the base (metres)

% Gripper pointing straight down: the tool's z-axis points along world -z.
R = [1 0 0; 0 -1 0; 0 0 -1];

% Starting guesses for the solver: the ready pose plus a few random poses.
rng(0);                                           % same random numbers on every run
qReady  = deg2rad([0, 15, 180, -130, 0, 55, 90]);
nRandom = 6;
seeds = zeros(nRandom + 1, 7);
seeds(1, :) = qReady;
for k = 1:nRandom
    seeds(k + 1, :) = robot.randomConfiguration;
end

ik = inverseKinematics("RigidBodyTree", robot);
ik.SolverParameters.AllowRandomRestart = false;   % do not search forever on impossible targets
ik.SolverParameters.MaxTime = 0.3;                % seconds per attempt
weights = [1 1 1 1 1 1];

okMap = false(numel(ys), numel(xs));
worstErr = 0;
fprintf("Testing %d positions (up to about a minute). One dot per position: ", numel(okMap));

for iy = 1:numel(ys)
    for ix = 1:numel(xs)
        tcp    = [xs(ix); ys(iy); tcpZ];          % wanted fingertip position
        flange = tcp - R * [0; 0; tcpOffsetM];    % where the end of the arm must be
        T = [R, flange; 0 0 0 1];

        bestDist = inf;
        for s = 1:size(seeds, 1)
            [q, info] = ik(eeName, T, weights, seeds(s, :));
            if ~strcmp(info.Status, "success")
                continue
            end
            [okJ, ~] = check_joint_limits(q, robot, safety);   % inside limits plus margin?
            if ~okJ
                continue
            end
            [okF, ~] = check_floor(robot, q, safety);          % nothing below the floor limit?
            if ~okF
                continue
            end
            dist = norm(q - qReady);              % prefer the solution closest to the ready pose
            if dist < bestDist
                bestDist = dist;
                bestQ = q;
            end
        end

        if isfinite(bestDist)
            okMap(iy, ix) = true;
            Tcheck   = getTransform(robot, bestQ, eeName);
            tcpCheck = Tcheck(1:3, 4) + Tcheck(1:3, 1:3) * [0; 0; tcpOffsetM];
            worstErr = max(worstErr, norm(tcpCheck - tcp));
        end
        fprintf(".");
    end
end

fprintf("\n\nFingertips %.0f mm above the ground, gripper pointing down.  OK = reachable safely\n\n", tcpZ * 1000);
fprintf("        x (forward, m):");
fprintf("%7.2f", xs);
fprintf("\n");
for iy = numel(ys):-1:1
    fprintf("y = %+5.2f              ", ys(iy));
    for ix = 1:numel(xs)
        if okMap(iy, ix)
            fprintf("     OK");
        else
            fprintf("     --");
        end
    end
    fprintf("\n");
end

fprintf("\n%d of %d positions are reachable.\n", nnz(okMap), numel(okMap));
if any(okMap(:))
    fprintf("Largest fingertip position error among them: %.4f mm\n", 1000 * worstErr);
end