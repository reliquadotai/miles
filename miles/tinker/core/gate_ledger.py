"""Per-rollout verdicts of the reliquary_gate loss, appended by the gateway (pay is computed from this record).

Rows are written at optim_step, once the gateway has judged the groups: each recorded rollout with its own verdict,
its group's verdict, whether it was backed out, and whether the step was applied. Forward-only passes, back-out
replays and steps never taken write nothing."""

import json
import math
import os
import time

GATE_LEDGER_ENV = "MILES_RELIQUARY_GATE_LEDGER"
GATE_FIELDS = ("admitted", "n_stoch", "m_stoch", "n_drawn", "m_drawn", "max_gap", "last_gap", "n_substitute")


def _finite(value):
    return value if not isinstance(value, float) or math.isfinite(value) else None


def record_step_verdicts(slot: int, rows: list[dict], outcome: dict) -> None:
    """``rows``: ``GateWindow.plan(...).rows``; ``outcome``: the slot's optim_step result (an ``error`` = not applied)."""
    path = os.environ.get(GATE_LEDGER_ENV)
    if not path or not rows:
        return
    now, stepped = time.time(), "error" not in outcome and not outcome.get("skipped_nonfinite")
    with open(path, "a") as handle:
        for row in rows:
            line = {"time": now, "slot": slot, "stepped": stepped, **{k: _finite(v) for k, v in row.items()}}
            handle.write(json.dumps(line, allow_nan=False) + "\n")
