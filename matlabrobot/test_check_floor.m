% test_check_floor.m
% Tests check_floor with the ready pose, using different floor heights.

robot  = loadrobot("kinovaGen3", "DataFormat", "row");
safety = safety_config();
q = deg2rad([0, 15, 180, -130, 0, 55, 90]);      % the ready pose

[ok, reason] = check_floor(robot, q, safety);
fprintf("1. floor at 10 mm  -> ok=%d | %s\n", ok, reason);

high = safety;
high.floorMinZ = 0.45;                            % pretend the floor is high
[ok, reason] = check_floor(robot, q, high);
fprintf("2. floor at 450 mm -> ok=%d | %s\n", ok, reason);

medium = safety;
medium.floorMinZ = 0.20;
[ok, reason] = check_floor(robot, q, medium);
fprintf("3. floor at 200 mm -> ok=%d | %s\n", ok, reason);