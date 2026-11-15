"""Reliquary's training gate inside a Miles Tinker loss (``reliquary_gate``): verification and loss in one forward.

Same rules as Reliquary's async trainer (``validator/training.py`` substitution loss + ``trainer/async_trainer.DrawGate``):

* per drawn position, on the fp32 head's logits: v* = argmax(logit/T + G(key, id)) over the true vocabulary, the
  Gumbel score gap of the miner's token (0 where it is v*), and the verifier's stochastic flag (max p < 0.99);
* the policy term scores v*, not the miner's token (substitution), with pi_old = the policy itself (ratio 1);
* a rollout stays in the loss when ``rollout_rate`` >= the floor, no gap is over the tolerance, and an EOS-ended
  rollout's last gap is within the terminal gap; otherwise it contributes nothing (but its verdict is reported);
* groups are judged by the caller once every rollout is forwarded (``judge_groups``): a rejected group's admitted
  rollouts are backed out with a second forward_backward at negated advantages, before optim_step.

The vocabulary may be split over the TP group: every reduction is a collective over identical rows.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from reliquary.constants import (
    FORCED_SEED_STOCHASTIC_MAXPROB,
    RL_FORGERY_MATCH_FLOOR,
    RL_GAP_TOLERANCE,
    RL_GROUP_MATCH_FLOOR,
    RL_ROLLOUT_MATCH_FLOOR,
    RL_SUBSTITUTE_GAP,
    RL_TERMINAL_GAP,
)
from reliquary.environment.forced_sampling import gumbel_noise
from reliquary.protocol.rl_draw import pooled_rate, rollout_rate

_NO_INDEX = torch.iinfo(torch.int64).max
_WORKSPACE_BYTES = 512 * 1024 * 1024
# per-datum verdict vector returned to the client (and logged by the gateway for pay)
GATE_FIELDS = ("admitted", "n_stoch", "m_stoch", "n_drawn", "m_drawn", "max_gap", "last_gap", "n_substitute")


def _all_reduce(tensor: torch.Tensor, op, group) -> torch.Tensor:
    if group is not None and dist.get_world_size(group) > 1:
        dist.all_reduce(tensor, op=op, group=group)
    return tensor


@torch.no_grad()
def gate_rows(
    logits: torch.Tensor,
    keys: torch.Tensor,
    targets: torch.Tensor,
    drawn: torch.Tensor,
    *,
    vocab_start: int,
    n_valid: int,
    group: dist.ProcessGroup | None = None,
    temperature: float = 1.0,
    stochastic_threshold: float = FORCED_SEED_STOCHASTIC_MAXPROB,
) -> dict[str, torch.Tensor]:
    """Per row of this rank's vocabulary shard ``logits`` [R, V_local] (columns >= ``n_valid`` are padding):
    ``vstar`` (int64; the miner's token where not drawn), ``gap`` (fp32, 0 where not drawn or matched),
    ``stochastic`` (bool) and ``match`` (bool: drawn and v* is the miner's token). Collective over ``group``."""
    rows, device = int(logits.shape[0]), logits.device
    keys = keys.to(device=device, dtype=torch.int64)
    targets = targets.to(device=device, dtype=torch.int64)
    drawn = drawn.to(device=device, dtype=torch.bool)
    best = torch.full((rows,), float("-inf"), dtype=torch.float32, device=device)
    pick = torch.full((rows,), _NO_INDEX, dtype=torch.int64, device=device)
    own = torch.full((rows,), float("-inf"), dtype=torch.float32, device=device)
    row_max = torch.full((rows,), float("-inf"), dtype=torch.float32, device=device)
    if rows and n_valid > 0:
        ids = torch.arange(vocab_start, vocab_start + n_valid, dtype=torch.int64, device=device)
        in_shard = (targets >= vocab_start) & (targets < vocab_start + n_valid)
        local_target = (targets - vocab_start).clamp(0, n_valid - 1)
        step = max(1, _WORKSPACE_BYTES // (n_valid * 8 * 6))
        for start in range(0, rows, step):
            end = min(rows, start + step)
            x = logits[start:end, :n_valid].detach().float()
            if temperature != 1.0:
                x = x / float(temperature)
            row_max[start:end] = x.max(dim=-1).values
            scores = x + gumbel_noise(keys[start:end], ids)
            chunk_best, chunk_arg = scores.max(dim=-1)  # first index on a tie
            best[start:end] = chunk_best
            pick[start:end] = chunk_arg + vocab_start
            mine = scores.gather(1, local_target[start:end, None])[:, 0]
            own[start:end] = torch.where(in_shard[start:end], mine, own[start:end])
    # v*: the max score, the lowest id on a tie across shards; the miner's score lives on one shard
    global_best = _all_reduce(best.clone(), dist.ReduceOp.MAX, group)
    candidate = torch.where(best == global_best, pick, torch.full_like(pick, _NO_INDEX))
    pick = _all_reduce(candidate, dist.ReduceOp.MIN, group)
    own = _all_reduce(own, dist.ReduceOp.MAX, group)
    # max p = 1 / sum exp(x - max) over the whole vocabulary
    global_max = _all_reduce(row_max, dist.ReduceOp.MAX, group)
    total = torch.zeros(rows, dtype=torch.float32, device=device)
    if rows and n_valid > 0:
        step = max(1, _WORKSPACE_BYTES // (n_valid * 4 * 3))
        for start in range(0, rows, step):
            end = min(rows, start + step)
            x = logits[start:end, :n_valid].detach().float()
            if temperature != 1.0:
                x = x / float(temperature)
            total[start:end] = (x - global_max[start:end, None]).exp().sum(dim=-1)
    total = _all_reduce(total, dist.ReduceOp.SUM, group)
    stochastic = total.reciprocal() < stochastic_threshold
    vstar = torch.where(drawn, pick, targets)
    match = drawn & (pick == targets)
    gap = torch.where(drawn & ~match, (global_best - own).clamp_min(0.0), torch.zeros_like(global_best))
    return {"vstar": vstar, "gap": gap, "stochastic": stochastic, "match": match}


def rollout_tally(match: torch.Tensor, stochastic: torch.Tensor, drawn: torch.Tensor) -> tuple[int, int, int, int]:
    """(stochastic, stochastic matches, drawn, drawn matches) over a rollout's positions (``rl_draw.rollout_rate``)."""
    stoch = drawn & stochastic
    return int(stoch.sum()), int((stoch & match).sum()), int(drawn.sum()), int((drawn & match).sum())


def rollout_suspects(gap, vstar, match, drawn, *, substitute_gap: float = RL_SUBSTITUTE_GAP, stop_ids=frozenset()):
    """(max gap, [(position, v*)] re-graded with the trainer's pick, last position's gap or None if not drawn)."""
    gaps, picks = gap.tolist(), vstar.tolist()
    drawn_l, match_l = drawn.tolist(), match.tolist()
    subs = [(i, int(picks[i])) for i in range(len(gaps))
            if gaps[i] > substitute_gap or (drawn_l[i] and not match_l[i] and int(picks[i]) in stop_ids)]
    last = None if not gaps or not drawn_l[-1] else float(gaps[-1])
    return (max(gaps, default=0.0), subs, last)


def rollout_verdict(tally, max_gap: float, last_gap: float | None, ends_on_eos: bool, *,
                    rollout_floor: float = RL_ROLLOUT_MATCH_FLOOR, gap_tolerance: float = RL_GAP_TOLERANCE,
                    terminal_gap: float = RL_TERMINAL_GAP) -> bool:
    """``DrawGate.admit`` for one rollout."""
    bad_terminal = ends_on_eos and last_gap is not None and last_gap > terminal_gap
    return rollout_rate(tally) >= rollout_floor and max_gap <= gap_tolerance and not bad_terminal


def judge_groups(groups, tallies, admitted, *, group_floor: float = RL_GROUP_MATCH_FLOOR,
                 forgery_floor: float = RL_FORGERY_MATCH_FLOOR) -> list[bool]:
    """``DrawGate.judge``'s acceptance: some rollout admitted, pooled match over the floor, none forged."""
    out = []
    for slots in groups:
        pooled = pooled_rate([tallies[s] for s in slots])
        forged = any(rollout_rate(tallies[s]) < forgery_floor for s in slots)
        out.append(any(admitted[s] for s in slots) and pooled >= group_floor and not forged)
    return out


def reliquary_gate_loss_function(args, batch, logits, sum_of_sample_mean):
    """Miles Tinker loss. Datum inputs: ``draw_keys`` (int64), ``draw_mask`` (1 where the miner drew), ``advantages``,
    ``logprobs`` (ignored: pi_old is the policy). Config: ``eos_ids`` ("a,b"), ``min_match`` (a client may only
    raise the rollout floor), ``temperature``."""
    from miles.backends.training_utils.loss.hub import tinker_losses as tl
    from miles.backends.training_utils.loss.hub.gumbel_match import true_vocab_size
    from miles.backends.training_utils.loss.hub.logit_processors import get_responses
    from miles.backends.training_utils.parallel import get_parallel_state

    config = batch.get("loss_fn_config") or {}
    if config.get("replay"):
        return _replay_loss(args, batch, logits)
    temperature = float(config.get("temperature", 1.0))
    floor = max(RL_ROLLOUT_MATCH_FLOOR, float(config.get("min_match", 0.0)))
    eos = {int(x) for x in str(config.get("eos_ids", "")).split(",") if x.strip()}
    parallel = get_parallel_state()
    tp_group = parallel.tp.group if parallel.tp.size > 1 else None
    width = int(logits.shape[-1])
    vocab_start = parallel.tp.rank * width if tp_group is not None else 0
    n_valid = min(max(true_vocab_size(args.hf_checkpoint) - vocab_start, 0), width)
    device = logits.device

    # 1) the draw on this forward's logits, per datum (this rank's CP slice)
    miner_targets = batch["target_tokens"]
    label_tokens = [
        torch.cat([tokens[: len(tokens) - len(targets)], tl._as_tensor_like(targets, tokens)])
        for tokens, targets in zip(batch["unconcat_tokens"], miner_targets, strict=True)
    ]
    local_keys = tl._local_response_values(args, batch, batch["draw_keys"])
    local_mask = tl._local_response_values(args, batch, batch["draw_mask"])
    local_targets = tl._local_response_values(args, batch, [list(t) for t in miner_targets])
    per = []
    for (chunk, _), keys, mask, targets in zip(
        get_responses(logits, args=args, unconcat_tokens=label_tokens, total_lengths=batch["total_lengths"],
                      response_lengths=batch["response_lengths"], max_seq_lens=batch.get("max_seq_lens", None)),
        local_keys, local_mask, local_targets, strict=True,
    ):
        per.append(gate_rows(
            chunk,
            torch.as_tensor(list(keys), dtype=torch.int64, device=device),
            torch.as_tensor(list(targets), dtype=torch.int64, device=device),
            torch.as_tensor(list(mask), device=device) > 0,
            vocab_start=vocab_start, n_valid=n_valid, group=tp_group, temperature=temperature,
        ))

    # 2) full-length rows on every CP rank (positions are zigzag-sliced), then per-datum verdicts
    lengths = list(batch["response_lengths"])
    if parallel.cp.size > 1:
        positions = tl._local_response_values(args, batch, [torch.arange(n, device=device) for n in lengths])
        packed = []
        for out, pos, n in zip(per, positions, lengths, strict=True):
            full = torch.zeros(n, 4, dtype=torch.float64, device=device)
            local = torch.stack([out["vstar"].double(), out["gap"].double(), out["stochastic"].double(),
                                 out["match"].double()], 1)
            packed.append(full.index_copy_(0, pos, local))
        flat = _all_reduce(torch.cat(packed) if packed else torch.zeros(0, 4, dtype=torch.float64, device=device),
                           dist.ReduceOp.SUM, parallel.cp.group)
        per = [{"vstar": p[:, 0].long(), "gap": p[:, 1].float(), "stochastic": p[:, 2] > 0, "match": p[:, 3] > 0}
               for p in flat.split(lengths)]
    verdicts = []
    for out, mask, targets in zip(per, batch["draw_mask"], miner_targets, strict=True):
        drawn = torch.as_tensor(list(mask), device=device) > 0
        tally = rollout_tally(out["match"], out["stochastic"], drawn)
        max_gap, subs, last = rollout_suspects(out["gap"], out["vstar"], out["match"], drawn, stop_ids=eos)
        ends_on_eos = bool(len(targets)) and int(targets[-1]) in eos
        ok = rollout_verdict(tally, max_gap, last, ends_on_eos, rollout_floor=floor)
        verdicts.append((ok, tally, max_gap, last, len(subs)))

    # 3) substitution loss: score v*, ratio 1 (pi_old = the policy), refused rollouts contribute nothing
    substituted = dict(batch)
    substituted["target_tokens"] = [out["vstar"].tolist() for out in per] if parallel.cp.size > 1 else [
        list(full) for full in _full_vstar(per, miner_targets, local_targets)]
    log_probs = tl._target_logprobs(args, substituted, logits)
    per_datum_losses = []
    for log_prob, advantage, mask, verdict in zip(
        log_probs, tl._local_response_values(args, batch, batch["advantages"]),
        tl._local_response_masks(args, batch, log_probs), verdicts, strict=True,
    ):
        ratio = torch.exp(log_prob - log_prob.detach())
        keep = 1.0 if verdict[0] else 0.0
        per_datum_losses.append(-(ratio * tl._as_tensor_like(advantage, log_prob) * mask).sum() * keep)
    loss, log = tl._sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)
    by_index = dict(zip((int(i) for i in batch["sample_indices"]), zip(per, verdicts)))
    for output in log.get("per_datum") or []:
        out, (ok, tally, max_gap, last, n_sub) = by_index[int(output["sample_index"])]
        output["vstar"] = out["vstar"].cpu()
        output["match"] = out["match"].float().cpu()
        output["gap"] = out["gap"].cpu()
        output["gate"] = torch.tensor([float(ok), *map(float, tally), max_gap, -1.0 if last is None else last,
                                       float(n_sub)])
    return loss, log


def _replay_loss(args, batch, logits):
    """A back-out pass: the client sends the first pass's v* as targets (negated advantages); no re-verification,
    so it cancels exactly what the first pass added. Same ratio-1 policy term."""
    from miles.backends.training_utils.loss.hub import tinker_losses as tl

    log_probs = tl._target_logprobs(args, batch, logits)
    per_datum_losses = [
        -(torch.exp(log_prob - log_prob.detach()) * tl._as_tensor_like(advantage, log_prob) * mask).sum()
        for log_prob, advantage, mask in zip(
            log_probs, tl._local_response_values(args, batch, batch["advantages"]),
            tl._local_response_masks(args, batch, log_probs), strict=True,
        )
    ]
    loss, log = tl._sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)
    for output in log.get("per_datum") or []:
        output["gate"] = torch.tensor([1.0, 0, 0, 0, 0, 0.0, -1.0, 0])
    return loss, log


def _full_vstar(per, miner_targets, local_targets):
    """CP 1: each rank holds whole responses; v* rows align with the full target list."""
    for out, full, local in zip(per, miner_targets, local_targets, strict=True):
        assert len(full) == len(local), "context parallel > 1 goes through the gathered path"
        yield out["vstar"].tolist()
