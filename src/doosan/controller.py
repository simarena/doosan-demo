"""Control strategies.

A controller receives the physics state the simulation just reported and
returns the next joint command, so the loop genuinely closes over the network.

**This is not a hard-realtime control path.** Browser → WebSocket → Python and
back carries millisecond-scale jitter plus occasional garbage-collection
pauses. That is fine for trajectory streaming, waypoints and supervisory
control. A stiff position loop at a few hundred hertz *will* oscillate: the
phase lag from transport delay eventually exceeds what gain margin you have.
If you need tight servo behaviour, the controller belongs inside the simulation
worker, with this server sending setpoints only.

Write your own by implementing :class:`Controller` — see
``examples/custom_controller.py``.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from .protocol import RobotState

#: Joint limits from the Doosan A0509 MJCF, radians. Joint 3 is the tightest.
JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-6.2832, 6.2832),
    (-6.2832, 6.2832),
    (-2.7925, 2.7925),
    (-6.2832, 6.2832),
    (-6.2832, 6.2832),
    (-6.2832, 6.2832),
)


def clamp(values: list[float]) -> list[float]:
    """Clamp a command to the arm's joint limits.

    Applied to every controller output. MuJoCo would clamp internally anyway,
    but silently — doing it here keeps the command we log identical to the one
    the robot executes.
    """
    out = []
    for i, v in enumerate(values):
        lo, hi = JOINT_LIMITS[i] if i < len(JOINT_LIMITS) else (-6.2832, 6.2832)
        out.append(max(lo, min(hi, v)))
    return out


@runtime_checkable
class Controller(Protocol):
    """Strategy interface.

    ``step`` is called once per received state message (~50 Hz by default), so
    control cadence follows the simulation rather than a local timer.
    """

    name: str

    def step(self, state: RobotState) -> list[float] | None:
        """Return joint targets in radians, or ``None`` to stay passive."""
        ...

    def reset(self) -> None:
        """Clear any accumulated internal state."""
        ...


class IdleController:
    """Sends nothing. The arm holds whatever setpoint it last received."""

    name = "idle"

    def step(self, state: RobotState) -> list[float] | None:
        return None

    def reset(self) -> None:
        pass


class SineController:
    """Smooth multi-joint sinusoid — a visible, safe default.

    Open-loop in intent: the trajectory is a function of simulation time. It
    still demonstrates a closed loop, because the time base comes from the
    state messages the simulation sends back.
    """

    name = "sine"

    def __init__(self, amplitude: float = 1.0, speed: float = 1.0) -> None:
        self.amplitude = amplitude
        self.speed = speed

    def step(self, state: RobotState) -> list[float] | None:
        if not state.n:
            return None
        t = state.sim_time * self.speed
        a = self.amplitude
        targets = [
            0.8 * a * math.sin(t * 0.5),
            0.4 * a * math.sin(t * 0.4) + 0.3,
            -0.6 + 0.3 * a * math.sin(t * 0.3),
            0.0,
            0.4 * a * math.sin(t * 0.6),
            0.0,
        ]
        return clamp(targets[: state.n])

    def reset(self) -> None:
        pass


class WaypointController:
    """Walks a list of joint-space poses, blending between them.

    Uses smoothstep rather than linear interpolation: a velocity discontinuity
    at each waypoint would show up as a torque spike, which then pollutes the
    wear signal you are trying to measure.
    """

    name = "waypoint"

    #: Reach, grasp, transfer, place, retreat — joints 2 and 3 carry most of
    #: the gravitational load, so they accumulate damage fastest.
    DEFAULT: tuple[tuple[list[float], float], ...] = (
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0),
        ([0.6, 0.5, -0.9, 0.0, 0.4, 0.0], 2.0),
        ([0.6, 0.8, -1.2, 0.0, 0.4, 0.0], 1.5),
        ([-0.8, 0.3, -0.7, 0.0, 0.4, 1.2], 2.5),
        ([-0.8, 0.8, -1.2, 0.0, 0.4, 1.2], 1.5),
    )

    def __init__(self, waypoints: tuple[tuple[list[float], float], ...] | None = None) -> None:
        self.waypoints = waypoints or self.DEFAULT
        self._t0: float | None = None

    @property
    def duration(self) -> float:
        return sum(d for _, d in self.waypoints)

    def step(self, state: RobotState) -> list[float] | None:
        if not state.n or not self.waypoints:
            return None

        # Anchor to the first state seen so the path starts at its beginning
        # regardless of how long the simulation has already been running.
        if self._t0 is None:
            self._t0 = state.sim_time

        # Segment i travels FROM waypoint i TO waypoint i+1, wrapping at the
        # end. Anchoring the segment at its start (rather than its target) is
        # what makes the path begin at waypoints[0] instead of jumping there
        # from the final pose.
        local = (state.sim_time - self._t0) % self.duration
        count = len(self.waypoints)
        for i, (start, dur) in enumerate(self.waypoints):
            if local < dur:
                target = self.waypoints[(i + 1) % count][0]
                x = local / dur
                a = x * x * (3 - 2 * x)  # smoothstep
                blended = [p + (q - p) * a for p, q in zip(start, target)]
                return clamp(blended[: state.n])
            local -= dur
        return clamp(list(self.waypoints[0][0])[: state.n])

    def reset(self) -> None:
        self._t0 = None


class PIDController:
    """Feedback controller tracking a joint-space setpoint.

    Gains are deliberately soft. Transport jitter means the measurement you
    integrate against is several milliseconds stale, so aggressive ``kp`` or
    any meaningful ``kd`` tends to amplify noise rather than reject it. Raise
    them only while watching for oscillation — and measure round-trip latency
    first with the ``ping`` command.
    """

    name = "pid"

    def __init__(
        self,
        target: list[float] | None = None,
        kp: float = 0.4,
        ki: float = 0.0,
        kd: float = 0.0,
        i_limit: float = 0.5,
    ) -> None:
        self.target = target or [0.0, 0.3, -0.6, 0.0, 0.2, 0.0]
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self._integral: list[float] = []
        self._last_t: float | None = None

    def step(self, state: RobotState) -> list[float] | None:
        n = state.n
        if not n:
            return None
        if len(self._integral) != n:
            self._integral = [0.0] * n

        dt = 0.0 if self._last_t is None else max(1e-4, state.sim_time - self._last_t)
        self._last_t = state.sim_time

        out = []
        for i in range(n):
            goal = self.target[i] if i < len(self.target) else 0.0
            err = goal - state.qpos[i]

            if self.ki and dt:
                self._integral[i] = max(
                    -self.i_limit, min(self.i_limit, self._integral[i] + err * dt)
                )

            # Derivative on measurement, not on error: a setpoint change would
            # otherwise produce a derivative kick the arm feels as a jolt.
            d = -state.qvel[i] if i < len(state.qvel) else 0.0

            out.append(goal + self.kp * err + self.ki * self._integral[i] + self.kd * d)
        return clamp(out)

    def reset(self) -> None:
        self._integral = []
        self._last_t = None


#: Controllers selectable by name from the CLI.
REGISTRY: dict[str, type] = {
    "idle": IdleController,
    "sine": SineController,
    "waypoint": WaypointController,
    "pid": PIDController,
}


def make(name: str) -> Controller:
    """Instantiate a registered controller by name."""
    return factory(name)()


def factory(name: str) -> type:
    """Look up a controller *class* by name.

    Prefer this over :func:`make` when handing a controller to the Hub: it
    builds one instance per robot, and sharing a single stateful object
    between arms would interleave their accumulated state.
    """
    if name not in REGISTRY:
        raise ValueError(f"unknown controller {name!r}; valid: {sorted(REGISTRY)}")
    return REGISTRY[name]
