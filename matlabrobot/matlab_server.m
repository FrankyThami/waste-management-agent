function matlab_server(port)
% matlab_server.m  (version 2)
% Listens on this laptop (127.0.0.1) for JSON commands from Python and
% drives ONE simulated arm (ArmSim) that remembers where it is.
% One JSON message per line in, one JSON reply per line out.
%
% Commands (JSON field names after the command name):
%   ping
%   get_state
%   check_position   x, y, z               (does not move)
%   move_to          x, y, z, animate      (metres; animate = true/false)
%   set_gripper      state                 ("open" or "closed")
%   set_boxes        boxes (a list)        (draws coloured table boxes; visual only)
%   stop                                   (halts; refuses moves until reset)
%   reset                                  (ready pose, gripper open)
%   quit                                   (stops the server)
%
% Stop it with Ctrl+C, or by sending "quit".

if nargin < 1
    port = 5005;
end

fprintf("Loading the simulated arm (a few seconds)...\n");
arm = ArmSim();

% 'server' is a local variable, so MATLAB closes the port automatically
% when this function ends (including Ctrl+C).
server = tcpserver("127.0.0.1", port, "Timeout", 5);
configureTerminator(server, "LF");
fprintf("MATLAB server listening on 127.0.0.1:%d  (press Ctrl+C to stop)\n", port);

running = true;
while running
    if server.NumBytesAvailable > 0
        line = readline(server);
        if isempty(line) || strlength(line) == 0
            continue
        end
        fprintf("[%s] received: %s\n", timestamp(), line);

        [reply, running] = handle_command(line, arm);

        replyText = jsonencode(reply);
        writeline(server, replyText);
        fprintf("[%s] replied : %s\n", timestamp(), replyText);
    else
        pause(0.05);    % nothing to do - rest a moment instead of spinning
    end
end
fprintf("MATLAB server stopped.\n");

end


function [reply, keepRunning] = handle_command(line, arm)
% Turns one JSON line into an action on the arm and returns the reply.
% All safety checks live inside ArmSim, so nothing here can bypass them.

keepRunning = true;

try
    msg = jsondecode(line);
catch
    reply = error_reply("Message is not valid JSON");
    return
end

if ~isstruct(msg) || ~isscalar(msg) || ~isfield(msg, "command")
    reply = error_reply("Message must be a JSON object with a 'command' field");
    return
end

try
    switch string(msg.command)
        case "ping"
            reply = struct("status", "ok", "reason", "pong");

        case "get_state"
            s = arm.getState();
            reply = struct("status", "ok", "reason", "OK", ...
                           "tip", s.tip, "gripper", s.gripper, "stopped", s.stopped, ...
                           "home", s.home);

        case "check_position"
            [pos, problem] = read_position(msg);
            if problem ~= ""
                reply = error_reply(problem);
            else
                reply = arm.checkPosition(pos);
            end

        case "move_to"
            [pos, problem] = read_position(msg);
            if problem ~= ""
                reply = error_reply(problem);
            else
                animate = true;
                if isfield(msg, "animate")
                    animate = logical(msg.animate);
                end
                reply = arm.moveTo(pos, animate);
            end

        case "set_gripper"
            if ~isfield(msg, "state")
                reply = error_reply("set_gripper needs a 'state' field");
            else
                reply = arm.setGripper(msg.state);
            end

        case "set_boxes"
            if ~isfield(msg, "boxes")
                reply = error_reply("set_boxes needs a 'boxes' field (a list, possibly empty)");
            else
                reply = arm.setBoxes(msg.boxes);
            end

        case "stop"
            reply = arm.stop();

        case "reset"
            reply = arm.reset();

        case "quit"
            reply = struct("status", "ok", "reason", "server stopping");
            keepRunning = false;

        otherwise
            reply = error_reply("Unknown command: " + string(msg.command));
    end
catch err
    reply = error_reply("MATLAB error: " + string(err.message));
end

end


function [pos, problem] = read_position(msg)
% Reads x, y, z (metres) from the message and checks they are single numbers.
pos = [];
problem = "";
if ~all(isfield(msg, {'x', 'y', 'z'}))
    problem = "Command needs 'x', 'y' and 'z' fields (metres)";
    return
end
values = {msg.x, msg.y, msg.z};
if ~all(cellfun(@(v) isnumeric(v) && isscalar(v), values))
    problem = "x, y and z must each be a single number";
    return
end
pos = [msg.x, msg.y, msg.z];
end


function reply = error_reply(reason)
reply = struct("status", "error", "reason", string(reason));
end


function t = timestamp()
t = char(datetime("now", "Format", "HH:mm:ss"));
end