function safety = safety_config()
% safety_config.m
% All safety settings live here. To change a limit, change it here only.
% NOTE: these are EXAMPLE values for the simulation. Later we will replace
% them with the real protection zone measured from the real arm.

% Layer 2: workspace box for the gripper tip (metres, robot base frame)
safety.box.xMin =  0.20;   safety.box.xMax = 0.70;
safety.box.yMin = -0.35;   safety.box.yMax = 0.35;
safety.box.zMin =  0.15;   safety.box.zMax = 0.65;   % zMin keeps the tip off the table

% Layer 1: stay this many degrees away from each joint's hard limit
% (we will use this in Step 1.3b)
safety.jointMarginDeg = 5;

% Layer 3: maximum tip speed in metres per second
% (we will use this in Step 1.4)
safety.maxTipSpeed = 0.10;

% Extra guard: biggest allowed change of any joint between two steps (degrees)
safety.maxJointStepDeg = 10;

% --- Floor-only safety profile (used from Stage G3) ---
safety.floorMinZ      = 0.010;   % nothing may go lower than this above the ground (metres)
safety.tcpOffsetM     = 0.15;    % EXAMPLE flange-to-fingertip length (metres) - measure the real one
safety.homeTcpHeightM = 0.050;   % fingertip height for the halt / observe / stop pose (metres)
safety.homeXY         = [0.30, 0.30];   % home position of the fingertips: [x y] in metres
end