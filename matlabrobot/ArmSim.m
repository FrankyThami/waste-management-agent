classdef ArmSim < handle
    % ArmSim.m  (version 2: gripper points DOWN, floor-only safety)
    % A simulated Kinova Gen3 that REMEMBERS where it is.
    %
    % Positions are FINGERTIP positions in metres, in the robot base frame
    % (x forward, y left, z up, z = 0 is the ground).
    %
    % Public methods (each returns a struct with the fields
    % status, reason, numPoints, maxTipSpeed, durationS):
    %   checkPosition(pos)    can the fingertips go to pos? (does not move)
    %   moveTo(pos, animate)  move the fingertips to pos in a straight line
    %   setGripper(state)     "open" or "closed" (visual only)
    %   setBoxes(boxes)       draw coloured boxes for the real table (visual only)
    %   stop()                halt; refuse moves until reset()
    %   reset()               back to the home pose, gripper open
    %   getState()            current fingertip position, gripper, stopped flag
    %
    % Safety (see safety_config.m): floor limit, joint limits with margin,
    % speed limit, sudden-joint-jump limit.

    properties
        robot                       % rigidBodyTree model of the Gen3
        safety                      % settings from safety_config()
        ikSolver                    % inverse kinematics solver
        eeName = "EndEffector_Link"
        qHome                       % home pose joint angles (radians)
        qNow                        % current joint angles (radians)
        orientation                 % 3x3 gripper direction: straight down, kept in every move
        extraSeeds                  % a few fixed starting guesses for the solver
        gripperState = "open"
        stopped = false
        dt = 0.05                   % seconds between simulation steps
        drawEvery = 2               % draw every 2nd point (bigger = faster animation)
        fig = []                    % figure handle (created on first animation)
        ax = []                     % axes handle
        pathLine = []               % red planned-path line
        gripperLine = []            % the drawn gripper (visual only)
        gripperLengthM = 0.15       % set from safety.tcpOffsetM in the constructor
        gripperOpenHalfM = 0.040    % EXAMPLE half-opening when open (metres)
        gripperClosedHalfM = 0.006  % EXAMPLE half-opening when closed (metres)
        toolAxis = [0 0 1]          % direction the fingers point (end-effector frame)
        spreadAxis = [0 1 0]        % direction the fingers open along
        boxHandles = {}             % the drawn table boxes (visual only)
    end

    methods
        function obj = ArmSim()
            obj.robot  = loadrobot("kinovaGen3", "DataFormat", "row");
            obj.safety = safety_config();
            obj.gripperLengthM = obj.safety.tcpOffsetM;
            obj.orientation = [1 0 0; 0 -1 0; 0 0 -1];     % tool z-axis points along world -z

            % Solver settings: no endless random restarts, short time limit
            solver = inverseKinematics("RigidBodyTree", obj.robot);
            solver.SolverParameters.AllowRandomRestart = false;
            solver.SolverParameters.MaxTime = 0.3;
            obj.ikSolver = solver;

            % A few fixed extra starting guesses (same every run)
            savedRng = rng;
            rng(0);
            obj.extraSeeds = zeros(6, 7);
            for k = 1:6
                obj.extraSeeds(k, :) = obj.robot.randomConfiguration;
            end
            rng(savedRng);

            [qH, ok, reason] = find_home_pose(obj.robot, obj.safety);
            if ~ok
                error("ArmSim:noHomePose", "Cannot find the home pose: %s", reason);
            end
            obj.qHome = qH;
            obj.reset();
        end

        function result = reset(obj)
            obj.qNow = obj.qHome;
            obj.gripperState = "open";
            obj.stopped = false;
            if obj.figureIsOpen()
                obj.drawPose(obj.qNow);
                obj.clearPathLine();
                obj.updateTitle();
            end
            result = obj.newResult();
            result.status = "done";
            result.reason = "Arm is at the home pose, gripper open";
        end

        function result = checkPosition(obj, pos)
            % Can the fingertips reach pos? Does NOT move the arm.
            % Tries several starting guesses for the solver, so a reachable
            % position is not rejected just because of one bad guess.
            result = obj.newResult();
            [pos, problem] = obj.cleanPosition(pos);
            if problem ~= ""
                result.reason = problem;
                return
            end
            problem = obj.floorProblem(pos);
            if problem ~= ""
                result.reason = problem;
                return
            end

            seeds = [obj.qNow; obj.qHome; obj.extraSeeds];
            lastReason = "";
            for s = 1:size(seeds, 1)
                [~, ok, reason] = obj.solve(pos, seeds(s, :));
                if ok
                    result.status = "ok";
                    result.reason = "OK";
                    return
                end
                lastReason = reason;
            end
            result.reason = lastReason;
        end

        function result = moveTo(obj, pos, animate)
            % Plan the WHOLE straight-line path and check every point first.
            % The arm only moves if every point passes.
            if nargin < 3
                animate = true;
            end
            result = obj.newResult();

            if obj.stopped
                result.reason = "Arm is stopped. Send 'reset' before moving again.";
                return
            end
            [pos, problem] = obj.cleanPosition(pos);
            if problem ~= ""
                result.reason = problem;
                return
            end
            problem = obj.floorProblem(pos);
            if problem ~= ""
                result.reason = problem;
                return
            end

            pos0 = obj.tipPosition();
            dist = norm(pos - pos0);
            if dist < 0.001
                result.status = "done";
                result.reason = "Already at the target";
                result.numPoints = 1;
                return
            end

            % Straight-line path of the fingertips, slow enough for the speed limit
            steps   = max(2, ceil(dist / (obj.safety.maxTipSpeed * obj.dt)));
            s       = linspace(0, 1, steps + 1)';
            tipPath = pos0 + s .* (pos - pos0);
            n       = size(tipPath, 1);

            qPath = zeros(n, 7);
            qPath(1, :) = obj.qNow;
            qGuess = obj.qNow;
            for i = 2:n
                [q, ok, reason] = obj.solve(tipPath(i, :), qGuess);
                if ~ok
                    result.reason = string(sprintf("Point %d: %s", i, reason));
                    return
                end
                stepDeg = max(abs(rad2deg(q - qPath(i - 1, :))));
                if stepDeg > obj.safety.maxJointStepDeg
                    result.reason = string(sprintf( ...
                        "Point %d: a joint would jump %.1f deg in one step (limit %.0f)", ...
                        i, stepDeg, obj.safety.maxJointStepDeg));
                    return
                end
                qPath(i, :) = q;
                qGuess = q;
            end

            % Planning passed - record the numbers, then move
            speeds = vecnorm(diff(tipPath), 2, 2) / obj.dt;
            result.numPoints   = n;
            result.maxTipSpeed = round(max(speeds), 4);
            result.durationS   = (n - 1) * obj.dt;

            if animate
                obj.playMotion(tipPath, qPath);
            elseif obj.figureIsOpen()
                obj.drawPose(qPath(end, :));
                drawnow;
            end

            obj.qNow = qPath(end, :);
            result.status = "done";
            result.reason = "OK";
        end

        function result = setGripper(obj, state)
            result = obj.newResult();
            if obj.stopped
                result.reason = "Arm is stopped. Send 'reset' first.";
                return
            end
            state = lower(string(state));
            if ~any(state == ["open", "closed"])
                result.reason = "Gripper state must be 'open' or 'closed'";
                return
            end
            obj.gripperState = state;
            if obj.figureIsOpen()
                obj.drawGripper(obj.qNow);
            end
            obj.updateTitle();
            result.status = "done";
            result.reason = "Gripper " + state;
        end

        function result = setBoxes(obj, boxes)
            % Replaces the boxes drawn on the table. VISUAL ONLY: the boxes do not
            % change any safety check or any move.
            % 'boxes' is a struct array, one element per box, with the fields
            %   name     text label, e.g. "red"
            %   x, y     centre of the box on the table (metres, robot base frame)
            %   yawDeg   turn about the vertical axis (degrees)
            %   lengthM  size along the box's own x direction (before turning)
            %   widthM   size along the box's own y direction (before turning)
            %   heightM  box height (the box stands on the table, z = 0)
            %   rgb      colour [r g b], each from 0 to 1
            % An empty list removes all boxes.
            result = obj.newResult();
            needed = ["name" "x" "y" "yawDeg" "lengthM" "widthM" "heightM" "rgb"];
            if isstruct(boxes) && ~all(isfield(boxes, needed))
                result.reason = "Each box needs the fields: " + strjoin(needed, ", ");
                return
            end
            obj.ensureFigure();
            obj.clearBoxes();
            count = 0;
            if isstruct(boxes)
                for k = 1:numel(boxes)
                    obj.drawBox(boxes(k));
                    count = count + 1;
                end
            end
            drawnow;
            result.status = "done";
            result.reason = sprintf("%d box(es) drawn", count);
        end

        function result = stop(obj)
            obj.stopped = true;
            obj.updateTitle();
            result = obj.newResult();
            result.status = "done";
            result.reason = "Arm stopped. Send 'reset' before moving again.";
        end

        function state = getState(obj)
            state = struct("tip", round(obj.tipPosition(), 4), ...
                           "gripper", obj.gripperState, ...
                           "stopped", obj.stopped, ...
                           "home", round(obj.tipPosition(obj.qHome), 4));
        end
    end

    methods (Access = private)
        function result = newResult(~)
            result = struct("status", "blocked", "reason", "", ...
                            "numPoints", 0, "maxTipSpeed", 0, "durationS", 0);
        end

        function [pos, problem] = cleanPosition(~, pos)
            problem = "";
            if ~isnumeric(pos) || numel(pos) ~= 3 || any(~isfinite(pos(:)))
                problem = "Position must be three finite numbers [x y z] in metres";
                return
            end
            pos = double(pos(:))';      % make sure it is a 1x3 row
        end

        function problem = floorProblem(obj, pos)
            % The one area rule: the fingertips may not go below the floor limit.
            problem = "";
            if pos(3) < obj.safety.floorMinZ
                problem = string(sprintf( ...
                    "Fingertips at z = %.1f mm are below the %.0f mm floor limit", ...
                    1000 * pos(3), 1000 * obj.safety.floorMinZ));
            end
        end

        function pos = tipPosition(obj, q)
            % Fingertip position = end of the arm + gripper length along the tool's z-axis.
            if nargin < 2
                q = obj.qNow;
            end
            T = getTransform(obj.robot, q, obj.eeName);
            pos = (T(1:3, 4) + T(1:3, 1:3) * [0; 0; obj.safety.tcpOffsetM])';
        end

        function [q, ok, reason] = solve(obj, pos, qGuess)
            % Inverse kinematics for the FINGERTIPS at pos (gripper pointing down),
            % then the floor check (fingertips and every link) and the joint-limit check.
            solver  = obj.ikSolver;
            flange  = pos(:) - obj.orientation * [0; 0; obj.safety.tcpOffsetM];
            Ttarget = [obj.orientation, flange; 0 0 0 1];
            [q, info] = solver(obj.eeName, Ttarget, [1 1 1 1 1 1], qGuess);
            ok = true;
            reason = "";
            if ~strcmp(info.Status, "success")
                ok = false;
                reason = string(sprintf("no exact joint solution for [%.3f %.3f %.3f] (error %.4f)", ...
                                        pos(1), pos(2), pos(3), info.PoseErrorNorm));
                return
            end
            [okF, reasonF] = check_floor(obj.robot, q, obj.safety);
            if ~okF
                ok = false;
                reason = string(reasonF);
                return
            end
            [okJ, reasonJ] = check_joint_limits(q, obj.robot, obj.safety);
            if ~okJ
                ok = false;
                reason = string(reasonJ);
            end
        end

        function tf = figureIsOpen(obj)
            tf = ~isempty(obj.fig) && isgraphics(obj.fig) && ...
                 ~isempty(obj.ax) && isgraphics(obj.ax);
        end

        function ensureFigure(obj)
            if obj.figureIsOpen()
                return
            end
            obj.fig = figure("Name", "Gen3 HAL simulation");
            obj.ax  = show(obj.robot, obj.qNow, "Frames", "off");
            hold(obj.ax, "on");

            % The floor limit: a see-through red plane
            fz = obj.safety.floorMinZ;
            patch(obj.ax, [-0.6 1 1 -0.6], [-0.7 -0.7 0.7 0.7], [fz fz fz fz], "r", ...
                  "FaceAlpha", 0.25, "EdgeColor", "none");

            view(obj.ax, 135, 20);
            obj.drawPose(obj.qNow);
            obj.updateTitle();
        end

        function drawPose(obj, q)
            show(obj.robot, q, "Parent", obj.ax, "PreservePlot", false, "Frames", "off");
            axis(obj.ax, [-0.5 1 -0.7 0.7 0 1]);
            obj.drawGripper(q);
        end

        function drawGripper(obj, q)
            % Draws a simple U-shaped gripper at the end of the arm (visual only).
            T = getTransform(obj.robot, q, obj.eeName);
            if obj.gripperState == "open"
                half = obj.gripperOpenHalfM;
                col  = [0 0.6 0];          % green
            else
                half = obj.gripperClosedHalfM;
                col  = [0.85 0 0];         % red
            end
            L = obj.gripperLengthM;

            % Four points of the U, in the end-effector's own frame:
            % left tip, left base, right base, right tip
            local = [ half * obj.spreadAxis + L * obj.toolAxis;
                      half * obj.spreadAxis;
                     -half * obj.spreadAxis;
                     -half * obj.spreadAxis + L * obj.toolAxis ];

            % Convert to world coordinates: rotate, then move
            world = (T(1:3, 1:3) * local')' + T(1:3, 4)';

            if isempty(obj.gripperLine) || ~isgraphics(obj.gripperLine)
                obj.gripperLine = plot3(obj.ax, world(:, 1), world(:, 2), world(:, 3), ...
                                        "-", "LineWidth", 4, "Color", col);
            else
                set(obj.gripperLine, "XData", world(:, 1), "YData", world(:, 2), ...
                    "ZData", world(:, 3), "Color", col);
            end
        end

        function clearBoxes(obj)
            % Removes every box (and its label) that setBoxes drew before.
            for k = 1:numel(obj.boxHandles)
                if isgraphics(obj.boxHandles{k})
                    delete(obj.boxHandles{k});
                end
            end
            obj.boxHandles = {};
        end

        function drawBox(obj, b)
            % One coloured box standing on the table, from z = 0 up to its height.
            half = [b.lengthM, b.widthM] / 2;
            corners = [-1 -1; 1 -1; 1 1; -1 1] .* half;      % 4 corners before turning
            a = deg2rad(b.yawDeg);
            turn = [cos(a) -sin(a); sin(a) cos(a)];
            xy = (turn * corners')' + [b.x, b.y];            % 4 corners on the table
            h = b.heightM;
            verts = [xy, zeros(4, 1); xy, h * ones(4, 1)];   % rows 1-4 bottom, rows 5-8 top
            faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
            rgb = double(b.rgb(:))';
            p = patch(obj.ax, "Vertices", verts, "Faces", faces, ...
                      "FaceColor", rgb, "FaceAlpha", 0.7, "EdgeColor", [0.2 0.2 0.2]);
            t = text(obj.ax, b.x, b.y, h + 0.015, char(b.name), ...
                     "HorizontalAlignment", "center", "FontSize", 8);
            obj.boxHandles{end + 1} = p;
            obj.boxHandles{end + 1} = t;
        end

        function clearPathLine(obj)
            if ~isempty(obj.pathLine) && isgraphics(obj.pathLine)
                delete(obj.pathLine);
            end
            obj.pathLine = [];
        end

        function playMotion(obj, tipPath, qPath)
            obj.ensureFigure();
            obj.clearPathLine();
            obj.pathLine = plot3(obj.ax, tipPath(:, 1), tipPath(:, 2), tipPath(:, 3), ...
                                 "r-", "LineWidth", 2);
            n = size(qPath, 1);
            frames = unique([1:obj.drawEvery:n, n]);
            for i = frames
                obj.drawPose(qPath(i, :));
                drawnow;
                pause(obj.dt);
            end
        end

        function updateTitle(obj)
            if ~obj.figureIsOpen()
                return
            end
            txt = "Gripper: " + upper(string(obj.gripperState));
            if obj.stopped
                txt = txt + "   |   ARM STOPPED";
            end
            title(obj.ax, txt);
            drawnow;
        end
    end
end