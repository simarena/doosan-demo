"""Telemetry receiver and remote controller for the SimArena Doosan A0509.

The browser runs MuJoCo; this package receives its physics and wear streams
and sends commands back, closing the control loop over a WebSocket.

    from doosan import Hub, SineController

    hub = Hub(controller=SineController())
    await hub.run(port=8787)
"""

from .controller import (
    Controller,
    IdleController,
    PIDController,
    SineController,
    WaypointController,
    make,
)
from .protocol import (
    ControlMode,
    JointReport,
    Pose,
    RobotState,
    StateEnvelope,
    TelemetryEnvelope,
)
from .recorder import Recorder
from .server import Hub

__all__ = [
    "Controller",
    "ControlMode",
    "Hub",
    "IdleController",
    "JointReport",
    "PIDController",
    "Pose",
    "Recorder",
    "RobotState",
    "SineController",
    "StateEnvelope",
    "TelemetryEnvelope",
    "WaypointController",
    "make",
]
