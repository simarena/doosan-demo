"""WebSocket server: telemetry receiver and remote controller in one process.

The browser connects here. Inbound ``state`` messages are fed to the active
controller, whose output goes straight back as ``ctrl`` — so the control loop
closes over the network rather than replaying a canned trajectory.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection, serve

from .controller import Controller, IdleController
from .protocol import (
    ControlMode,
    JointReport,
    StateEnvelope,
    TelemetryEnvelope,
    cmd_ctrl,
    cmd_echo,
    cmd_mode,
)
from .recorder import Recorder


def _as_factory(
    controller: Controller | Callable[[], Controller] | None,
) -> Callable[[], Controller]:
    """Normalise a controller argument into a zero-arg factory.

    A *class* and an *instance* both expose ``step``, so `hasattr` cannot tell
    them apart — `isinstance(c, type)` can. Classes and factories are called
    per robot; a bare instance is wrapped and shared, which is only safe for
    stateless controllers.
    """
    if controller is None:
        return IdleController
    if isinstance(controller, type) or not hasattr(controller, "step"):
        return controller  # type: ignore[return-value]
    return lambda: controller  # type: ignore[return-value]


class Hub:
    """Connection registry, message router and control loop.

    One hub serves any number of simulations, though in practice there is one
    browser tab. Commands broadcast to every connected client.
    """

    def __init__(
        self,
        controller: Controller | Callable[[], Controller] | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        self.clients: set[ServerConnection] = set()
        self.recorder = recorder or Recorder(None)

        # Several arms may share one physics world, and WaypointController and
        # PIDController both carry per-arm state. The hub therefore holds a
        # FACTORY and builds one controller per robot — sharing a single
        # object would interleave its internals between arms.
        #
        # A plain instance is still accepted for the single-arm case; it is
        # then reused for every robot, which is only safe for stateless
        # controllers like Idle and Sine.
        self._factory: Callable[[], Controller] = _as_factory(controller)
        self.controllers: dict[str, Controller] = {}

        #: Loop runs only while True AND the sim has granted remote authority.
        self.loop_enabled = False
        #: Control authority per robot, as last reported by the simulation.
        self.modes: dict[str, ControlMode] = {}

        self.latest: StateEnvelope | None = None
        self.latest_wear: list[JointReport] = []

        self.states = 0
        self.batches = 0
        self.records = 0
        self.commands = 0
        self.rtt_ms: float | None = None

        self._pings: dict[int, float] = {}
        self._ping_id = 0
        self._on_event: Callable[[str], None] | None = None

    def on_event(self, fn: Callable[[str], None]) -> None:
        """Register a log sink, so the CLI can render activity."""
        self._on_event = fn

    def log(self, msg: str) -> None:
        if self._on_event:
            self._on_event(msg)

    # ── outbound ──────────────────────────────────────────────────────────────

    async def send(self, msg: dict[str, Any]) -> None:
        """Broadcast to every connected simulation."""
        if not self.clients:
            self.log("[warn] no simulation connected")
            return
        payload = json.dumps(msg)
        await asyncio.gather(*(c.send(payload) for c in self.clients), return_exceptions=True)

    @property
    def controller(self) -> Controller:
        """Representative controller, for display. One exists per robot."""
        return next(iter(self.controllers.values()), None) or self._factory()

    @controller.setter
    def controller(self, c: Controller | Callable[[], Controller]) -> None:
        """Swap the strategy for every robot.

        Accepts an instance (reused across arms — safe only if stateless) or a
        factory, which is preferred since most controllers carry per-arm state.
        """
        self._factory = _as_factory(c)
        self.controllers = {}

    def controller_for(self, robot: str) -> Controller:
        """The controller driving one robot, created on first use."""
        if robot not in self.controllers:
            self.controllers[robot] = self._factory()
        return self.controllers[robot]

    async def set_mode(self, mode: ControlMode, robot: str | None = None) -> None:
        """Claim or release control authority. Omit `robot` for all of them.

        Releasing also disengages the loop: continuing to compute commands the
        simulation would reject is just a source of confusing logs.
        """
        targets = [robot] if robot is not None else (self.robots or [None])
        for name in targets:
            if name is not None:
                self.modes[name] = mode
            await self.send(cmd_mode(mode, name))
        if mode != "remote":
            self.loop_enabled = False
        self.log(f"[mode] {mode} → {robot or 'all'}")

    async def engage(self, on: bool, robot: str | None = None) -> None:
        """Start or stop closed-loop control, claiming authority as needed."""
        if on:
            await self.set_mode("remote", robot)
            # Give the sim a moment to apply the mode before commands arrive,
            # otherwise the first few hit its authority guard and are dropped.
            await asyncio.sleep(0.1)
            for name in [robot] if robot is not None else self.robots:
                self.controller_for(name).reset()
        self.loop_enabled = on
        self.log(f"[loop] {self.controller.name} {'engaged' if on else 'released'}")

    @property
    def robots(self) -> list[str]:
        """Robots seen in the most recent state message."""
        return self.latest.names if self.latest else []

    async def ping(self) -> None:
        """Measure round-trip latency."""
        self._ping_id += 1
        self._pings[self._ping_id] = time.perf_counter()
        await self.send(cmd_echo(self._ping_id))

    # ── inbound ───────────────────────────────────────────────────────────────

    async def handle(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self.log("[warn] non-JSON message ignored")
            return

        kind = msg.get("type")
        if kind == "state":
            await self._on_state(StateEnvelope.from_dict(msg))
        elif kind == "telemetry":
            self._on_telemetry(TelemetryEnvelope.from_dict(msg))
        elif kind == "echo":
            started = self._pings.pop(msg.get("id"), None)
            if started is not None:
                self.rtt_ms = (time.perf_counter() - started) * 1000
                self.log(f"[rtt] {self.rtt_ms:.1f}ms")
        else:
            self.log(f"[warn] unknown message type {kind!r}")

    async def _on_state(self, state: StateEnvelope) -> None:
        self.latest = state
        self.states += 1

        # Track authority as the sim reports it, per robot — the user may flip
        # a mode in the browser UI, and the loop must not fight that.
        for name, robot in state.robots.items():
            self.modes[name] = robot.mode

        if not self.loop_enabled:
            return

        # Every robot that granted authority gets a command this tick. Arms
        # still in 'scripted' are skipped, so the two can run side by side.
        granted = [r for r in state.robots.values() if r.mode == "remote"]
        if not granted:
            return

        for robot in granted:
            try:
                command = self.controller_for(robot.name).step(robot)
            except Exception as exc:  # noqa: BLE001 - a bad controller must not kill the server
                self.loop_enabled = False
                self.log(f"[error] controller for {robot.name!r} raised: {exc!r}")
                return
            if command is None:
                continue
            self.commands += 1
            await self.send(cmd_ctrl(command, robot.name))

    def _on_telemetry(self, env: TelemetryEnvelope) -> None:
        self.batches += 1
        self.records += len(env.joints)
        if env.joints:
            self.latest_wear = env.joints
        self.recorder.write(env)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def _serve_client(self, ws: ServerConnection) -> None:
        self.clients.add(ws)
        self.log(f"[+] simulation connected ({len(self.clients)} active)")
        try:
            async for raw in ws:
                await self.handle(raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            self.loop_enabled = False
            self.log(f"[-] simulation disconnected ({len(self.clients)} active)")

    async def run(self, host: str = "localhost", port: int = 8787) -> None:
        async with serve(self._serve_client, host, port):
            await asyncio.Future()
