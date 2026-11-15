"""The gateway's side of Reliquary's group back-out: groups are judged and backed out at optim_step, not by clients.

Every ``reliquary_gate`` forward_backward is recorded per slot (datums, verdicts, v*) until the slot's optim_step.
There, groups (``group_ids``) are judged on the recorded verdicts (``judge_groups``: pooled match, forgery floor);
each recorded batch holding an admitted rollout of a rejected group is replayed whole, in the same order (the same
packing keeps the numerics, so the back-out cancels to the bf16 floor), with the back-out rollouts at negated
advantages, every other rollout at zero, all scored on the first pass's v* (``replay``: no re-verification).
Clients cannot send ``replay`` themselves, and cannot step past a rejected group.

Limit: with multi-LoRA a batch may pack several slots; only this slot's datums are replayed (other packing, larger
residual). Full training has one slot, so its replays are exact.
"""

from dataclasses import dataclass, field

GATE_FIELDS = ("admitted", "n_stoch", "m_stoch", "n_drawn", "m_drawn", "max_gap", "last_gap", "n_substitute")


@dataclass
class _Batch:
    batch_id: int
    datums: list  # this slot's datums, in batch order
    outputs: list[dict]
    config: dict


@dataclass
class BackOutPlan:
    groups: int = 0
    rejected: int = 0
    backed_out: int = 0
    replays: list = field(default_factory=list)  # (batch_id, [(slot, datum)], loss_fn_config)


class GateWindow:
    def __init__(self) -> None:
        self._batches: dict[int, list[_Batch]] = {}

    def record(self, slot: int, batch_id: int, slot_datums: list, outputs: list[dict], config: dict | None) -> None:
        config = dict(config or {})
        if config.get("replay"):
            return
        rows = [(datum, output) for (s, datum), output in zip(slot_datums, outputs, strict=True)
                if s == slot and not datum.get("padding")]
        if rows:
            self._batches.setdefault(slot, []).append(
                _Batch(batch_id, [d for d, _ in rows], [o for _, o in rows], config))

    def clear(self, slot: int) -> None:
        self._batches.pop(slot, None)

    def plan(self, slot: int, judge=None) -> BackOutPlan:
        if judge is None:
            from miles.backends.training_utils.loss.hub.reliquary_gate import judge_groups as judge
        batches = self._batches.get(slot, [])
        rows = [(b, i) for b in batches for i in range(len(b.datums))]
        if not rows:
            return BackOutPlan()
        verdicts = [dict(zip(GATE_FIELDS, b.outputs[i]["gate"], strict=True)) for b, i in rows]
        admitted = [bool(v["admitted"]) for v in verdicts]
        tallies = [tuple(int(v[k]) for k in ("n_stoch", "m_stoch", "n_drawn", "m_drawn")) for v in verdicts]
        by_group: dict[int, list[int]] = {}
        for n, (b, i) in enumerate(rows):
            by_group.setdefault(int(b.datums[i]["group_ids"][0]), []).append(n)
        groups = list(by_group.values())
        accepted = judge(groups, tallies, admitted)
        back = {n for g, ok in zip(groups, accepted, strict=True) if not ok for n in g if admitted[n]}
        plan = BackOutPlan(groups=len(groups), rejected=accepted.count(False), backed_out=len(back))
        n = 0
        for b in batches:
            marks = [n + i in back for i in range(len(b.datums))]
            n += len(b.datums)
            if any(marks):
                replay = [(slot, _backed_out(d, o, m)) for d, o, m in zip(b.datums, b.outputs, marks, strict=True)]
                plan.replays.append((b.batch_id, replay, b.config | {"replay": 1}))
        return plan


def _backed_out(datum: dict, output: dict, back: bool) -> dict:
    """``datum`` at negated advantages (or zero), scored on the first pass's v*."""
    advantages = [-float(a) if back else 0.0 for a in datum["advantages"]]
    return datum | {"advantages": advantages, "target_tokens": [int(v) for v in output["vstar"]]}
