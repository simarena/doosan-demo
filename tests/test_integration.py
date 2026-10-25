"""End-to-end check over a real WebSocket.

Stands in for the browser: sends exactly the payloads ``bridge.ts`` emits and
asserts the server closes the loop correctly. Catches the integration bugs
that unit tests structurally cannot.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from doosan.controller import SineController
from doosan.server import Hub

PORT = 8899


def robot_payload(qpos: list[float], mode: str) -> dict:
    return {
        "mode": mode,
        "qpos": qpos,
        "qvel": [0.0] * len(qpos),
        "ctrl": qpos,
        "torque": [10.0] * len(qpos),
        "ee": {"pos": [0.4, 0.0, 0.6], "quat": [1.0, 0.0, 0.0, 0.0]},
    }


def state_msg(t: float, qpos: list[float], mode: str = "remote", **others: str) -> str:
    """A payload shaped exactly like the frontend's StateEnvelope.

    `others` adds further robots as name=mode, so a test can mix authority
    across arms sharing the world.
    """
    robots = {"armA": robot_payload(qpos, mode)}
    for name, m in others.items():
        robots[name] = robot_payload(qpos, m)
    return json.dumps(
        {
            "type": "state",
            "seq": 0,
            "sent": 0,
            "simTime": t,
            "session": "itest",
            "scene": "cell",
            "robots": robots,
        }
    )


class Sim:
    """Minimal stand-in for the browser side of the bridge."""

    def __init__(self, ws):
        self.ws = ws
        self.received: list[dict] = []
        self._task = asyncio.create_task(self._drain())

    async def _drain(self):
        try:
            async for raw in self.ws:
                self.received.append(json.loads(raw))
        except websockets.ConnectionClosed:
            pass

    def of_type(self, kind: str) -> list[dict]:
        return [m for m in self.received if m.get("type") == kind]

    async def send(self, payload: str):
        await self.ws.send(payload)

    def close(self):
        self._task.cancel()


@pytest.fixture
async def hub_and_sim():
    # A factory, not an instance: stateful controllers must not be shared
    # between arms, and the Hub builds one per robot from this.
    hub = Hub(controller=SineController)
    server = asyncio.create_task(hub.run("localhost", PORT))
    await asyncio.sleep(0.3)

    async with websockets.connect(f"ws://localhost:{PORT}") as ws:
        sim = Sim(ws)
        await asyncio.sleep(0.1)
        try:
            yield hub, sim
        finally:
            sim.close()

    server.cancel()
    try:
        await server
    except asyncio.CancelledError:
        pass


async def test_telemetry_is_ingested(hub_and_sim):
    hub, sim = hub_and_sim
    await sim.send(
        json.dumps(
            {
                "type": "telemetry",
                "seq": 0,
                "sent": 0,
                "simTime": 1.0,
                "session": "itest",
                "scene": "cell",
                "joints": [
                    {"joint": j, "robot": "armA", "damage": 1e-5 * j, "frictionRatio": 1.2}
                    for j in range(1, 7)
                ],
            }
        )
    )
    await asyncio.sleep(0.2)
    assert hub.batches == 1
    assert hub.records == 6
    assert hub.latest_wear[1].friction == 1.2


async def test_state_is_ingested(hub_and_sim):
    hub, sim = hub_and_sim
    await sim.send(state_msg(1.0, [0.1] * 6))
    await asyncio.sleep(0.2)
    assert hub.states == 1
    assert hub.latest is not None
    assert hub.latest.names == ["armA"]
    assert hub.latest.robots["armA"].n == 6
    assert hub.latest.robots["armA"].ee is not None


async def test_loop_closes_over_the_network(hub_and_sim):
    hub, sim = hub_and_sim
    await hub.engage(True)

    for i in range(5):
        await sim.send(state_msg(i * 0.2, [0.05 * i] * 6))
        await asyncio.sleep(0.08)
    await asyncio.sleep(0.2)

    ctrls = sim.of_type("ctrl")
    assert len(ctrls) >= 3, "server should answer state with ctrl"
    assert len(ctrls[0]["values"]) == 6
    # Commands must track the evolving state, not repeat a canned sequence.
    assert ctrls[0]["values"] != ctrls[-1]["values"]


async def test_no_commands_without_authority(hub_and_sim):
    """The guard that stops the server fighting a local trajectory.

    With per-robot authority the loop stays armed — other arms may still be
    remote — but an arm reporting 'scripted' must receive nothing.
    """
    hub, sim = hub_and_sim
    await hub.engage(True)
    sim.received.clear()

    await sim.send(state_msg(1.0, [0.1] * 6, mode="scripted"))
    await asyncio.sleep(0.25)

    assert hub.modes["armA"] == "scripted"
    assert sim.of_type("ctrl") == []


async def test_commands_only_the_arm_that_granted_authority(hub_and_sim):
    """Mixed authority: armA remote, armB scripted, in one shared world."""
    hub, sim = hub_and_sim
    await hub.engage(True)
    sim.received.clear()

    for i in range(4):
        await sim.send(state_msg(i * 0.2, [0.05 * i] * 6, mode="remote", armB="scripted"))
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.2)

    ctrls = sim.of_type("ctrl")
    assert ctrls, "the remote arm should be driven"
    targets = {c.get("robot") for c in ctrls}
    assert targets == {"armA"}, f"only armA should be commanded, got {targets}"


async def test_drives_several_arms_independently(hub_and_sim):
    """Both arms remote — each gets its own addressed command."""
    hub, sim = hub_and_sim
    await hub.engage(True)
    sim.received.clear()

    for i in range(4):
        await sim.send(state_msg(i * 0.2, [0.05 * i] * 6, mode="remote", armB="remote"))
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.2)

    targets = {c.get("robot") for c in sim.of_type("ctrl")}
    assert targets == {"armA", "armB"}, f"expected both arms, got {targets}"


async def test_each_arm_gets_its_own_controller(hub_and_sim):
    """Stateful controllers must not share accumulation between arms."""
    hub, sim = hub_and_sim
    await hub.engage(True)
    await sim.send(state_msg(1.0, [0.1] * 6, mode="remote", armB="remote"))
    await asyncio.sleep(0.25)

    assert set(hub.controllers) == {"armA", "armB"}
    assert hub.controllers["armA"] is not hub.controllers["armB"]


async def test_releasing_authority_stops_the_loop(hub_and_sim):
    hub, sim = hub_and_sim
    await hub.engage(True)
    await sim.send(state_msg(1.0, [0.1] * 6))
    await asyncio.sleep(0.15)

    await hub.set_mode("scripted")
    sim.received.clear()
    await sim.send(state_msg(2.0, [0.2] * 6, mode="scripted"))
    await asyncio.sleep(0.2)

    assert sim.of_type("ctrl") == []


async def test_echo_measures_latency(hub_and_sim):
    hub, sim = hub_and_sim
    await hub.ping()
    await asyncio.sleep(0.15)

    probe = sim.of_type("echo")
    assert probe, "server should emit an echo probe"
    await sim.send(json.dumps({"type": "echo", "id": probe[0]["id"]}))
    await asyncio.sleep(0.15)

    assert hub.rtt_ms is not None
    assert hub.rtt_ms >= 0


async def test_controller_crash_disengages_loop(hub_and_sim):
    """A broken controller must not take the server down with it."""
    hub, sim = hub_and_sim

    class Broken:
        name = "broken"

        def step(self, state):
            raise RuntimeError("boom")

        def reset(self):
            pass

    hub.controller = Broken()
    await hub.engage(True)
    await sim.send(state_msg(1.0, [0.1] * 6))
    await asyncio.sleep(0.2)

    assert hub.loop_enabled is False
    assert hub.clients, "connection should survive a controller error"


async def test_malformed_message_is_ignored(hub_and_sim):
    hub, sim = hub_and_sim
    await sim.send("not json at all")
    await asyncio.sleep(0.15)
    assert hub.clients, "a bad frame must not drop the connection"
