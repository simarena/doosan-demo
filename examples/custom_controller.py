#!/usr/bin/env python
"""Writing your own controller.

    uv run python examples/custom_controller.py

Then connect the browser at /play/doosan. This script claims control
authority as soon as a simulation appears, so no typed commands are needed.
"""

from __future__ import annotations

import asyncio
import math

from doosan import Hub, RobotState
from doosan.controller import clamp


class CircleController:
    """Sweeps joint 1 while nodding joints 2 and 3 in quadrature.

    Shows the three things a real controller needs:

    1. reading measured state (``state.qpos``) rather than assuming position
    2. a feedforward target plus a *gentle* correction — the transport is
       jittery, so a stiff gain here oscillates rather than tracks
    3. clamping, so a bug cannot command past a joint limit
    """

    name = "circle"

    def __init__(self, period: float = 8.0, radius: float = 0.7) -> None:
        self.period = period
        self.radius = radius
        self._t0: float | None = None

    def step(self, state: RobotState) -> list[float] | None:
        if not state.n:
            return None

        if self._t0 is None:
            self._t0 = state.sim_time

        phase = 2 * math.pi * (state.sim_time - self._t0) / self.period

        target = [
            self.radius * math.sin(phase),
            0.35 + 0.25 * math.cos(phase),
            -0.7 + 0.25 * math.sin(phase),
            0.0,
            0.3 * math.cos(phase),
            0.0,
        ][: state.n]

        # Proportional trim against the position the simulation reported.
        # 0.2 is already on the assertive side for this transport.
        kp = 0.2
        corrected = [q + kp * (q - state.qpos[i]) for i, q in enumerate(target)]
        return clamp(corrected)

    def reset(self) -> None:
        self._t0 = None


class LoggingController:
    """Wraps another controller and reports the degradation gap.

    ``state.ctrl`` is what the actuator actually received — after delay,
    backlash and lost motion. Comparing it with what we asked for is the
    cleanest way to see wear affecting control.
    """

    def __init__(self, inner, every: int = 50) -> None:
        self.inner = inner
        self.every = every
        self.name = f"logged:{inner.name}"
        self._n = 0
        self._last: list[float] | None = None

    def step(self, state: RobotState) -> list[float] | None:
        if self._last is not None and self._n % self.every == 0:
            gap = max(abs(a - b) for a, b in zip(self._last, state.ctrl))
            err = max(abs(a - b) for a, b in zip(state.ctrl, state.qpos))
            print(f"t={state.sim_time:7.2f}  applied-gap={gap:.4f}  tracking-err={err:.4f}")

        self._n += 1
        out = self.inner.step(state)
        if out is not None:
            self._last = out
        return out

    def reset(self) -> None:
        self._n = 0
        self._last = None
        self.inner.reset()


async def main() -> None:
    # A factory, so each arm in the scene gets its own controller instance.
    hub = Hub(controller=lambda: LoggingController(CircleController()))
    hub.on_event(print)

    async def engage_when_connected() -> None:
        """Claim authority as soon as the browser shows up."""
        while not hub.clients:
            await asyncio.sleep(0.2)
        await asyncio.sleep(0.3)  # let the first state arrive
        await hub.engage(True)

    print("listening on ws://localhost:8787 — open /play/doosan and press Connect")
    await asyncio.gather(hub.run(port=8787), engage_when_connected())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nstopped")
