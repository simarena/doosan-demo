"""Wire protocol for the SimArena Doosan A0509 bridge.

Mirrors ``src/lib/sim/bridge.ts`` in the frontend. The browser runs MuJoCo and
is the *source* of physics; this package is the sink for telemetry and the
source of commands.

Two independent inbound streams:

==============  ======  =================================================
stream          rate    carries
==============  ======  =================================================
``telemetry``   10 Hz   per-joint wear and damage accumulation
``state``       50 Hz   qpos, qvel, ctrl, torque, end-effector pose
==============  ======  =================================================

Outbound commands: ``mode``, ``ctrl``, ``reset``, ``stop``, ``start``,
``wear``, ``echo``, ``configure``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ControlMode = Literal["scripted", "remote", "idle"]
MODES: tuple[ControlMode, ...] = ("scripted", "remote", "idle")


# ──────────────────────────────────────────────────────────────────────────────
# Field-name asymmetry
# ──────────────────────────────────────────────────────────────────────────────
#
# The protocol is NOT symmetric: outbound telemetry uses operator-facing names
# while inbound wear patches use the simulator's internal names. Rather than
# leak that into user code, this package exposes the short (patch) names
# everywhere and translates at the boundary.
#
# If the frontend protocol is ever made symmetric, delete this map and the
# translation in JointReport.from_dict — nothing else depends on it.
REPORT_TO_PARAM: dict[str, str] = {
    "frictionRatio": "friction",
    "wearSlop": "slop",
    "lostMotion": "lostMotion",
    "motorStrength": "strength",
    "actuatorDelay": "delay",
    "mountCompliance": "compliance",
    "hoursSinceJointReplacement": "hoursSinceReplacement",
    "hoursSinceRegrease": "hoursSinceRegrease",
    "checkoutTemp": "checkoutTemp",
}

PARAM_TO_REPORT: dict[str, str] = {v: k for k, v in REPORT_TO_PARAM.items()}

#: Degradation parameters accepted by a ``wear`` command, in patch-name form.
WEAR_PARAMS: tuple[str, ...] = (
    "friction",
    "slop",
    "lostMotion",
    "strength",
    "delay",
    "compliance",
    "hoursSinceReplacement",
    "hoursSinceRegrease",
    "checkoutTemp",
)


@dataclass(slots=True)
class JointReport:
    """One joint's wear record at one sampled tick.

    Splits three ways, which matters for interpreting it: *inputs* configure a
    worn robot, *static* fields are fleet bookkeeping the simulation never
    reads, and *derived* fields are integrated from observed physics. Only the
    derived ones are actually measured.
    """

    joint: int
    #: Owning robot, from the MuJoCo name prefix. "" when un-namespaced.
    robot: str = ""

    # Input — fed to physics
    friction: float = 1.0
    slop: float = 0.0
    lost_motion: float = 0.0
    strength: float = 1.0
    delay: float = 0.0
    compliance: float = 0.0

    # Static — logged, never simulated
    hours_since_replacement: float = 0.0
    hours_since_regrease: float = 0.0
    checkout_temp: float = 20.0

    # Derived — integrated from simulation
    wear_acceleration: float = 0.0
    highest_damage: float = 0.0
    damage: float = 0.0
    travel: float = 0.0
    reversals: int = 0
    work: float = 0.0
    peak_torque: float = 0.0
    duty: float = 0.0

    # Instantaneous
    pos: float = 0.0
    vel: float = 0.0
    torque: float = 0.0
    tracking_error: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JointReport:
        return cls(
            joint=int(d.get("joint", 0)),
            robot=str(d.get("robot", "")),
            friction=float(d.get("frictionRatio", 1.0)),
            slop=float(d.get("wearSlop", 0.0)),
            lost_motion=float(d.get("lostMotion", 0.0)),
            strength=float(d.get("motorStrength", 1.0)),
            delay=float(d.get("actuatorDelay", 0.0)),
            compliance=float(d.get("mountCompliance", 0.0)),
            hours_since_replacement=float(d.get("hoursSinceJointReplacement", 0.0)),
            hours_since_regrease=float(d.get("hoursSinceRegrease", 0.0)),
            checkout_temp=float(d.get("checkoutTemp", 20.0)),
            wear_acceleration=float(d.get("wearAcceleration", 0.0)),
            highest_damage=float(d.get("highestDamage", 0.0)),
            damage=float(d.get("damage", 0.0)),
            travel=float(d.get("travel", 0.0)),
            reversals=int(d.get("reversals", 0)),
            work=float(d.get("work", 0.0)),
            peak_torque=float(d.get("peakTorque", 0.0)),
            duty=float(d.get("duty", 0.0)),
            pos=float(d.get("pos", 0.0)),
            vel=float(d.get("vel", 0.0)),
            torque=float(d.get("torque", 0.0)),
            tracking_error=float(d.get("trackingError", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Flat record for a JSONL log, using snake_case keys."""
        return {
            "joint": self.joint,
            "robot": self.robot,
            "friction": self.friction,
            "slop": self.slop,
            "lost_motion": self.lost_motion,
            "strength": self.strength,
            "delay": self.delay,
            "compliance": self.compliance,
            "hours_since_replacement": self.hours_since_replacement,
            "hours_since_regrease": self.hours_since_regrease,
            "checkout_temp": self.checkout_temp,
            "wear_acceleration": self.wear_acceleration,
            "highest_damage": self.highest_damage,
            "damage": self.damage,
            "travel": self.travel,
            "reversals": self.reversals,
            "work": self.work,
            "peak_torque": self.peak_torque,
            "duty": self.duty,
            "pos": self.pos,
            "vel": self.vel,
            "torque": self.torque,
            "tracking_error": self.tracking_error,
        }


@dataclass(slots=True)
class Pose:
    """End-effector world pose. Quaternion is MuJoCo order ``[w, x, y, z]``."""

    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Pose | None:
        if not d:
            return None
        p = d.get("pos") or [0.0, 0.0, 0.0]
        q = d.get("quat") or [1.0, 0.0, 0.0, 0.0]
        return cls(
            pos=(float(p[0]), float(p[1]), float(p[2])),
            quat=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        )


@dataclass(slots=True)
class RobotState:
    """One robot's physics state within a frame."""

    name: str
    mode: ControlMode
    qpos: list[float]
    qvel: list[float]
    #: Command actually applied, i.e. AFTER delay/backlash/lost-motion.
    #: Comparing it against your last command reveals the degradation gap.
    ctrl: list[float]
    torque: list[float]
    ee: Pose | None = None
    #: Copied from the envelope so a controller needs only this object.
    sim_time: float = 0.0

    @property
    def n(self) -> int:
        """Number of actuated joints."""
        return len(self.qpos)

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any], sim_time: float = 0.0) -> RobotState:
        return cls(
            name=name,
            sim_time=sim_time,
            mode=d.get("mode", "idle"),
            qpos=[float(v) for v in d.get("qpos", [])],
            qvel=[float(v) for v in d.get("qvel", [])],
            ctrl=[float(v) for v in d.get("ctrl", [])],
            torque=[float(v) for v in d.get("torque", [])],
            ee=Pose.from_dict(d.get("ee")),
        )


@dataclass(slots=True)
class StateEnvelope:
    """Physics state at one instant, for every robot sharing the world.

    Several arms may occupy one scene and collide with each other, so state is
    keyed by robot. A single un-namespaced robot appears under the "" key.
    """

    seq: int
    sent: int
    sim_time: float
    session: str
    scene: str
    robots: dict[str, RobotState]

    @property
    def names(self) -> list[str]:
        return list(self.robots)

    def robot(self, name: str) -> RobotState | None:
        """One robot's state, or None if absent this frame."""
        return self.robots.get(name)

    @property
    def only(self) -> RobotState | None:
        """The sole robot, for single-arm scenes. None if ambiguous."""
        return next(iter(self.robots.values())) if len(self.robots) == 1 else None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StateEnvelope:
        return cls(
            seq=int(d.get("seq", 0)),
            sent=int(d.get("sent", 0)),
            sim_time=float(d.get("simTime", 0.0)),
            session=str(d.get("session", "")),
            scene=str(d.get("scene", "")),
            robots={
                name: RobotState.from_dict(name, r, float(d.get("simTime", 0.0)))
                for name, r in (d.get("robots") or {}).items()
            },
        )


@dataclass(slots=True)
class TelemetryEnvelope:
    """A batch of wear records, one per joint per sampled tick."""

    seq: int
    sent: int
    sim_time: float
    session: str
    #: Scene identifier; each record carries its own ``robot``.
    scene: str
    joints: list[JointReport] = field(default_factory=list)

    def for_robot(self, name: str) -> list[JointReport]:
        return [j for j in self.joints if j.robot == name]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TelemetryEnvelope:
        return cls(
            seq=int(d.get("seq", 0)),
            sent=int(d.get("sent", 0)),
            sim_time=float(d.get("simTime", 0.0)),
            session=str(d.get("session", "")),
            scene=str(d.get("scene", "")),
            joints=[JointReport.from_dict(j) for j in d.get("joints", [])],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Outbound commands
# ──────────────────────────────────────────────────────────────────────────────


def cmd_mode(mode: ControlMode, robot: str | None = None) -> dict[str, Any]:
    """Claim or release control authority, per robot.

    Authority is held independently by each robot, so one arm can run its
    local scripted trajectory while this server drives another. Omit ``robot``
    to address every arm at once.

    The sim rejects ``ctrl`` unless that robot's mode is ``remote`` — a
    deliberate guard so a stray command cannot fight a scripted trajectory.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    out: dict[str, Any] = {"type": "mode", "mode": mode}
    if robot is not None:
        out["robot"] = robot
    return out


def cmd_ctrl(values: list[float], robot: str | None = None) -> dict[str, Any]:
    """Joint position targets in radians. Requires ``remote`` mode.

    Values are 1-based per robot, resolved to actuators by name rather than by
    position, so a scene reordering cannot silently misroute them.
    """
    out: dict[str, Any] = {"type": "ctrl", "values": [float(v) for v in values]}
    if robot is not None:
        out["robot"] = robot
    return out


def cmd_wear(joint: int, robot: str | None = None, **params: float) -> dict[str, Any]:
    """Patch degradation parameters for one joint.

    Accepts the short names in :data:`WEAR_PARAMS`; unknown keys raise rather
    than being silently dropped on the wire. Omit ``robot`` to patch that joint
    index on every arm.
    """
    unknown = set(params) - set(WEAR_PARAMS)
    if unknown:
        raise ValueError(
            f"unknown wear params {sorted(unknown)}; valid: {list(WEAR_PARAMS)}"
        )
    patch: dict[str, Any] = {"joint": int(joint), **params}
    if robot is not None:
        patch["robot"] = robot
    return {"type": "wear", "joints": [patch]}


def cmd_reset() -> dict[str, Any]:
    return {"type": "reset"}


def cmd_stop() -> dict[str, Any]:
    return {"type": "stop"}


def cmd_start() -> dict[str, Any]:
    return {"type": "start"}


def cmd_echo(ident: int) -> dict[str, Any]:
    """Latency probe. The sim echoes it straight back."""
    return {"type": "echo", "id": int(ident)}


def cmd_configure(
    state_hz: int | None = None, telemetry_hz: int | None = None
) -> dict[str, Any]:
    """Adjust stream rates.

    Above roughly 100 Hz the browser's timer jitter dominates, so raising
    ``state_hz`` further costs bandwidth without buying real precision.
    """
    out: dict[str, Any] = {"type": "configure"}
    if state_hz is not None:
        out["stateHz"] = int(state_hz)
    if telemetry_hz is not None:
        out["telemetryHz"] = int(telemetry_hz)
    return out
