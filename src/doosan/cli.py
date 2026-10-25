"""Command-line entry point with a live dashboard.

    uv run doosan-server
    uv run doosan-server --port 8787 --log run.jsonl --controller sine

Type ``help`` at the prompt for commands.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .controller import REGISTRY, factory
from .protocol import MODES, cmd_ctrl, cmd_reset, cmd_start, cmd_stop, cmd_wear
from .recorder import Recorder
from .server import Hub

console = Console()
events: deque[str] = deque(maxlen=8)

HELP = (
    "[bold]commands[/bold]\n"
    "  loop on|off [robot]        engage closed-loop control\n"
    f"  controller <name>          strategy: {' '.join(sorted(REGISTRY))}\n"
    "  mode <mode> [robot]        set control authority (per robot)\n"
    "  ctrl [robot] <q1>..<q6>    one-shot joint command (radians)\n"
    "  wear <j> <key> <v> [robot] patch degradation, e.g. wear 2 friction 3.1\n"
    "  start | stop | reset       transport control\n"
    "  ping                       measure round-trip latency\n"
    "  quit                       exit\n"
)


def _numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def dashboard(hub: Hub) -> Group:
    """Render the live view."""
    head = Table.grid(padding=(0, 2))
    modes = " ".join(f"{n or 'robot'}:{m}" for n, m in hub.modes.items()) or "—"
    head.add_row(
        f"[bold]modes[/] {modes}",
        f"[bold]loop[/] {'[green]on' if hub.loop_enabled else '[dim]off'}[/]",
        f"[bold]controller[/] {hub.controller.name}",
        f"[bold]rtt[/] {f'{hub.rtt_ms:.1f}ms' if hub.rtt_ms else '—'}",
        f"[bold]clients[/] {len(hub.clients)}",
    )

    counts = Table.grid(padding=(0, 2))
    counts.add_row(
        f"[dim]states[/] {hub.states}",
        f"[dim]batches[/] {hub.batches}",
        f"[dim]records[/] {hub.records}",
        f"[dim]commands[/] {hub.commands}",
        f"[dim]logged[/] {hub.recorder.lines}" if hub.recorder.active else "[dim]not logging[/]",
    )

    joints = Table(expand=True, header_style="dim")
    joints.add_column("robot")
    joints.add_column("J")
    for col in ("qpos", "qvel", "ctrl", "torque", "damage", "wear accel"):
        joints.add_column(col, justify="right")

    state = hub.latest
    wear = {(w.robot, w.joint): w for w in hub.latest_wear}
    rows = 0
    for name, robot in (state.robots if state else {}).items():
        for i in range(robot.n):
            w = wear.get((name, i + 1))
            label = name or "—"
            joints.add_row(
                f"[dim]{label}[/]" if i else f"{label} [dim]{robot.mode}[/]",
                str(i + 1),
                f"{robot.qpos[i]:+.3f}",
                f"{robot.qvel[i]:+.3f}",
                f"{robot.ctrl[i]:+.3f}",
                f"{robot.torque[i]:+.1f}",
                f"{w.damage:.2e}" if w else "—",
                f"{w.wear_acceleration:+.2e}" if w else "—",
            )
            rows += 1
    if not rows:
        joints.add_row("[dim]waiting for simulation…[/]", "", "", "", "", "", "", "")

    ee = ""
    if state:
        parts = [
            f"{n or 'robot'} {r.ee.pos[0]:+.2f},{r.ee.pos[1]:+.2f},{r.ee.pos[2]:+.2f}"
            for n, r in state.robots.items()
            if r.ee
        ]
        if parts:
            ee = "  [dim]ee[/] " + "   ".join(parts)

    log = "\n".join(events) or "[dim]no events yet[/]"

    return Group(
        Panel(Group(head, counts), title="doosan", border_style="blue"),
        Panel(joints, title=f"joints{ee}", border_style="dim"),
        Panel(log, title="events", border_style="dim"),
    )


async def dispatch(hub: Hub, cmd: str, rest: list[str]) -> bool:
    """Execute one typed command. Returns False to quit."""
    if cmd in ("quit", "exit"):
        return False

    if cmd == "help":
        console.print(HELP)
    elif cmd == "loop":
        on = rest[0] != "off" if rest else True
        robot = rest[1] if len(rest) > 1 else None
        await hub.engage(on, robot)
    elif cmd == "controller":
        if not rest or rest[0] not in REGISTRY:
            hub.log(f"[warn] usage: controller {'|'.join(sorted(REGISTRY))}")
            return True
        was_running = hub.loop_enabled
        hub.controller = factory(rest[0])
        hub.log(f"[controller] {hub.controller.name}")
        if was_running:
            await hub.engage(True)
    elif cmd == "mode":
        # mode <scripted|remote|idle> [robot]
        if not rest or rest[0] not in MODES:
            hub.log(f"[warn] usage: mode {'|'.join(MODES)} [robot]")
            return True
        await hub.set_mode(rest[0], rest[1] if len(rest) > 1 else None)  # type: ignore[arg-type]
    elif cmd == "ctrl":
        # ctrl [robot] <q1> <q2> ... — a leading non-numeric token is a robot.
        args = list(rest)
        robot = None
        if args and not _numeric(args[0]):
            robot = args.pop(0)
        values = [float(v) for v in args]
        if not values:
            hub.log("[warn] usage: ctrl [robot] <q1> <q2> ... (radians)")
            return True
        await hub.send(cmd_ctrl(values, robot))
    elif cmd == "wear":
        # wear <joint> <key> <value> [robot]
        if len(rest) not in (3, 4):
            hub.log("[warn] usage: wear <joint> <key> <value> [robot]")
            return True
        robot = rest[3] if len(rest) == 4 else None
        await hub.send(cmd_wear(int(rest[0]), robot, **{rest[1]: float(rest[2])}))
    elif cmd == "start":
        await hub.send(cmd_start())
    elif cmd == "stop":
        await hub.send(cmd_stop())
    elif cmd == "reset":
        hub.controller.reset()
        await hub.send(cmd_reset())
    elif cmd == "ping":
        await hub.ping()
    elif cmd == "state":
        if not hub.latest:
            hub.log("[warn] no state received yet")
            return True
        fmt = lambda a: " ".join(f"{v:+.3f}" for v in a)  # noqa: E731
        for name, r in hub.latest.robots.items():
            console.print(
                f"\n  [bold]{name or 'robot'}[/]  {r.mode}  t={hub.latest.sim_time:.2f}s"
            )
            console.print(f"    qpos   {fmt(r.qpos)}")
            console.print(f"    ctrl   {fmt(r.ctrl)}  (post-degradation)")
            console.print(f"    torque {fmt(r.torque)}")
            if r.ee:
                console.print(f"    ee     {fmt(r.ee.pos)}")
    else:
        hub.log(f"[warn] unknown command {cmd!r} — try 'help'")
    return True


async def repl(hub: Hub) -> None:
    """Read commands from stdin without blocking the event loop."""
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return
        parts = line.strip().split()
        if not parts:
            continue
        try:
            if not await dispatch(hub, parts[0], parts[1:]):
                return
        except (ValueError, IndexError) as exc:
            hub.log(f"[warn] {exc}")


async def _refresh(live: Live, hub: Hub) -> None:
    while True:
        live.update(dashboard(hub))
        await asyncio.sleep(0.125)


async def run(args: argparse.Namespace) -> None:
    recorder = Recorder(args.log)
    hub = Hub(controller=factory(args.controller), recorder=recorder)
    hub.on_event(events.append)

    hub.log(f"listening on ws://{args.host}:{args.port}")
    if recorder.active:
        hub.log(f"logging to {recorder.path}")
    hub.log("type 'help' for commands")

    server = asyncio.create_task(hub.run(args.host, args.port))
    try:
        with Live(dashboard(hub), console=console, refresh_per_second=8) as live:
            refresher = asyncio.create_task(_refresh(live, hub))
            try:
                await repl(hub)
            finally:
                refresher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await refresher
    finally:
        server.cancel()
        recorder.close()
        with contextlib.suppress(asyncio.CancelledError):
            await server


def main() -> None:
    p = argparse.ArgumentParser(prog="doosan-server", description=__doc__)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--log", default=None, help="append telemetry to a JSONL file")
    p.add_argument(
        "--controller", default="idle", choices=sorted(REGISTRY), help="initial strategy"
    )
    args = p.parse_args()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run(args))
    console.print("\n[dim]stopped[/]")


if __name__ == "__main__":
    main()
