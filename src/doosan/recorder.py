"""JSONL telemetry log.

One line per joint per sampled tick, flattened so the file loads directly into
pandas or DuckDB without unnesting:

    df = pd.read_json("run.jsonl", lines=True)
    df.groupby("joint").damage.max()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

from .protocol import TelemetryEnvelope


class Recorder:
    """Appends telemetry to a JSONL file. No-op when no path is given."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._fh: TextIO | None = None
        self.lines = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    @property
    def active(self) -> bool:
        return self._fh is not None

    def write(self, env: TelemetryEnvelope) -> None:
        if not self._fh:
            return
        for joint in env.joints:
            record = {
                "session": env.session,
                "robot": env.robot,
                "sim_time": env.sim_time,
                "wall_time": env.sent,
                **joint.to_dict(),
            }
            self._fh.write(json.dumps(record) + "\n")
            self.lines += 1
        # Flushed per batch so a killed process still leaves a usable log;
        # at 10 Hz the syscall cost is irrelevant.
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
