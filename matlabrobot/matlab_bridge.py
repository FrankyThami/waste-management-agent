"""
matlab_bridge.py
Client that talks to the MATLAB simulation server (matlab_server.m)
over a local TCP connection. Messages are JSON, one per line.
Uses only the Python standard library, so nothing needs installing.
"""
import json
import socket


class MatlabBridgeError(Exception):
    """Raised when we cannot talk to the MATLAB server."""


class MatlabRobotClient:
    def __init__(self, host="127.0.0.1", port=5005, timeout_s=60.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s      # how long to wait for MATLAB to answer
        self._sock = None
        self._file = None

    # ---------- connection handling ----------
    def connect(self):
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=5.0)
        except OSError as error:
            raise MatlabBridgeError(
                f"Cannot connect to MATLAB at {self.host}:{self.port}. "
                "Is matlab_server running in MATLAB?"
            ) from error
        self._sock.settimeout(self.timeout_s)
        self._file = self._sock.makefile("rw", encoding="utf-8", newline="\n")

    def close(self):
        for item in (self._file, self._sock):
            try:
                if item is not None:
                    item.close()
            except OSError:
                pass
        self._file = None
        self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # ---------- sending one message and reading one reply ----------
    def _send(self, message):
        if self._file is None:
            raise MatlabBridgeError("Not connected. Call connect() first.")
        try:
            self._file.write(json.dumps(message) + "\n")
            self._file.flush()
            line = self._file.readline()
        except OSError as error:        # includes timeouts
            self.close()
            raise MatlabBridgeError(f"Communication with MATLAB failed: {error}") from error

        if not line:
            self.close()
            raise MatlabBridgeError("MATLAB closed the connection.")
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise MatlabBridgeError(f"MATLAB sent an unreadable reply: {line!r}") from error

    # ---------- the commands ----------
    def ping(self):
        """Check that MATLAB is alive."""
        return self._send({"command": "ping"})

    def get_state(self):
        """Where is the tip, is the gripper open or closed, is the arm stopped?"""
        return self._send({"command": "get_state"})

    def check_position(self, x, y, z):
        """Ask MATLAB whether the tip could go to (x, y, z) metres. Does not move."""
        return self._send({"command": "check_position",
                           "x": float(x), "y": float(y), "z": float(z)})

    def move_to(self, x, y, z, animate=True):
        """Move the simulated tip to (x, y, z) metres in a straight line."""
        return self._send({"command": "move_to",
                           "x": float(x), "y": float(y), "z": float(z),
                           "animate": bool(animate)})

    def set_gripper(self, state):
        """state is "open" or "closed" (drawn in the MATLAB window)."""
        return self._send({"command": "set_gripper", "state": str(state)})

    def set_boxes(self, boxes):
        """Draw coloured boxes on the simulated table (visual only). An empty list clears them."""
        return self._send({"command": "set_boxes", "boxes": list(boxes)})

    def stop(self):
        """Halt the arm. It refuses moves until reset() is called."""
        return self._send({"command": "stop"})

    def reset(self):
        """Back to the ready pose, gripper open, stop cleared."""
        return self._send({"command": "reset"})

    def quit_server(self):
        """Ask the MATLAB server to stop."""
        return self._send({"command": "quit"})