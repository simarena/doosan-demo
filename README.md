# doosan

Telemetry receiver **and** remote controller for the SimArena Doosan A0509 wear
simulation — one process doing both.

The browser runs MuJoCo and owns the physics. This package receives its state
and wear streams, and sends commands back, closing the control loop over a
WebSocket.

```
┌──────────────────┐   state 50Hz, telemetry 10Hz   ┌──────────────────┐
│  browser         │ ─────────────────────────────▶ │  doosan-server   │
│  /play/doosan    │                                 │  (this package)  │
│  MuJoCo physics  │ ◀───────────────────────────── │  controller      │
└──────────────────┘   ctrl, mode, wear, reset      └──────────────────┘
```

## Quickstart

```bash
cd /Users/ted/Downloads/doosan
uv run doosan-server
```

Then open `/play/doosan` in the frontend and press **Connect**. In the server
prompt:

```
loop on
```

The arm starts moving under server control. Commands are typed at the prompt;
`help` lists them.

```bash
uv run doosan-server --controller sine --log run.jsonl --port 8787
```

## Commands

| command | effect |
|---|---|
| `loop on [robot]` / `loop off` | engage closed-loop control (claims authority automatically) |
| `controller <name>` | switch strategy: `idle`, `sine`, `waypoint`, `pid` |
| `mode <mode> [robot]` | set control authority, per robot |
| `ctrl [robot] <q1> … <q6>` | one-shot joint command, radians |
| `wear <joint> <key> <value> [robot]` | patch degradation, e.g. `wear 2 friction 3.1 armB` |
| `state` | print the latest physics state for every arm |
| `start` / `stop` / `reset` | transport control |
| `ping` | measure round-trip latency |
| `quit` | exit |

## Control authority

The simulation enforces a single writer for `data.ctrl`:

- **`scripted`** — the browser's local trajectory drives the arm (default)
- **`remote`** — this server drives it; the local trajectory stands down
- **`idle`** — nobody drives; the arm holds its last setpoint

`ctrl` commands are **rejected unless the mode is `remote`**. `loop on` claims
authority for you. If someone flips the mode in the browser UI, the server
notices and disengages rather than fighting it.

## Multiple robots

A scene may hold several arms sharing one physics world — they collide and can
interact. Each is identified by its MuJoCo name prefix (`armA`, `armB`), which
the scene format assigns when the same asset is instanced twice.

Everything is per robot:

- **Authority** — `mode remote armA` drives one arm while the other keeps
  running its local scripted trajectory.
- **State** — `state.robots["armA"]`, each with its own joints and mode.
- **Wear** — every `JointReport` carries `robot`.
- **Commands** — `ctrl armB 0.3 0.5 …`, resolved to actuators *by name*, so a
  scene reordering cannot silently misroute them.

A single un-namespaced robot appears under the `""` key, so one arm is just
N=1 and needs no special handling.

## Writing a controller

Implement one method:

```python
from doosan import Hub, RobotState

class MyController:
    name = "mine"

    def step(self, state: RobotState) -> list[float] | None:
        # state.qpos, qvel, ctrl, torque — one entry per joint
        # state.ee.pos / .quat — end-effector world pose
        # state.name — which arm this is
        return [0.0] * state.n          # or None to stay passive

    def reset(self) -> None:
        pass

# Pass the CLASS or a factory, not an instance: the Hub builds one controller
# per robot, and sharing a stateful object would interleave the arms.
hub = Hub(controller=MyController)
```

`step` is called once per robot per received state message (~50 Hz), so
control cadence follows the simulation rather than a local timer. Outputs are
clamped to the arm's joint limits. An exception inside `step` disengages the
loop instead of killing the server.

See `examples/custom_controller.py`.

## Reading the state

`state.ctrl` is the command **actually applied** — after the simulated
actuator delay, backlash dead-zone and lost motion. Comparing it against what
you last sent shows the degradation gap directly, which is the point of the
wear model.

`state.ee.quat` is MuJoCo order `[w, x, y, z]`, not `[x, y, z, w]`.

## Telemetry log

`--log run.jsonl` writes one flat line per joint per tick:

```python
import pandas as pd
df = pd.read_json("run.jsonl", lines=True)
df.groupby("joint").damage.max()
```

Fields split three ways, and conflating them is the main way this gets
misread:

- **Input** (`friction`, `slop`, `lost_motion`, `strength`, `delay`,
  `compliance`) — configure a worn robot; fed to physics.
- **Static** (`hours_since_replacement`, `hours_since_regrease`,
  `checkout_temp`) — fleet bookkeeping the simulation never reads.
- **Derived** (`damage`, `wear_acceleration`, `highest_damage`, `travel`,
  `reversals`, `work`, `peak_torque`, `duty`) — integrated from observed
  physics. **Only these are actually measured.**

## Limitations

**This is not a hard-realtime control path.** Browser → WebSocket → Python and
back carries millisecond-scale jitter plus occasional GC pauses. Fine for
trajectory streaming, waypoints and supervisory control. A stiff position loop
at a few hundred hertz *will* oscillate — transport phase lag eventually
exceeds your gain margin. Run `ping` to measure actual round-trip before
trusting anything tight. If you need a real servo loop, the controller belongs
inside the simulation worker with this server sending setpoints only.

`PIDController` ships with deliberately soft gains for that reason.

**Field names are asymmetric on the wire.** Outbound telemetry uses
operator-facing names (`frictionRatio`), inbound patches use internal ones
(`friction`). This package hides that — you use the short names everywhere —
but it matters if you read raw JSON. See `REPORT_TO_PARAM` in `protocol.py`.

**Backlash is a control-side approximation.** The simulation models wear slop
as a torque dead-zone, not a physical gear gap, so the joint will not rattle
under gravity the way worn hardware does.

## Development

```bash
uv run pytest
```
