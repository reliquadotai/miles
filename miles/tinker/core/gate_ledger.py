"""Per-rollout verdicts of the reliquary_gate loss, appended by the gateway (pay is computed from this record)."""

import json
import os
import time

GATE_LEDGER_ENV = "MILES_RELIQUARY_GATE_LEDGER"
GATE_FIELDS = ("admitted", "n_stoch", "m_stoch", "n_drawn", "m_drawn", "max_gap", "last_gap", "n_substitute")


def record_gate_verdicts(batch_id: int, slot_datums: list, outputs: list[dict], loss_fn_config: dict | None = None) -> None:
    """The gateway's own record of every rollout verdict (miners are paid from it, never from the client).
    A back-out replay re-scores rollouts already recorded: it is not a verdict."""
    path = os.environ.get(GATE_LEDGER_ENV)
    if not path or (loss_fn_config or {}).get("replay"):
        return
    with open(path, "a") as handle:
        for index, ((slot, _), output) in enumerate(zip(slot_datums, outputs, strict=True)):
            row = dict(zip(GATE_FIELDS, output["gate"], strict=True))
            row["admitted"] = bool(row["admitted"])
            handle.write(json.dumps({"time": time.time(), "batch_id": batch_id, "slot": slot, "index": index, **row}) + "\n")
