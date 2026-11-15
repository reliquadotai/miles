"""Reliquary's forced-draw check inside a Miles Tinker loss.

``gumbel_match`` = ``importance_sampling`` plus two per-datum outputs computed from the same forward:

* ``vstar``: at every label position, ``argmax_v (logit_v / T + G_v(key_t))`` over the full true vocabulary, with
  Reliquary's float32 Gumbel noise (``reliquary.environment.forced_sampling``: splitmix64 of (key, global token id));
* ``match``: 1.0 where ``vstar == target_token``, else 0.0.

The client sends ``draw_keys`` (int64, one per label position: ``gumbel_key(u_t)`` of the rollout's public uniform,
anything on prompt / tool positions, which carry weight 0) next to ``logprobs`` and ``advantages``.

Megatron splits the vocabulary over the TP group: each rank scores its own shard with the GLOBAL token ids, drops the
padding columns past the true vocabulary, and the winner is the max score over the group, lowest id on a tie (the
unsplit ``torch.argmax`` rule). Scores are elementwise, so the result is bit-identical to the unsplit reference.
"""
from __future__ import annotations

import functools
import json
import os

import torch
import torch.distributed as dist

from reliquary.environment.forced_sampling import gumbel_noise

_NO_INDEX = torch.iinfo(torch.int64).max
# int64/float32 [rows, vocab] temporaries of the torch noise (~6 live int64 copies).
_WORKSPACE_BYTES = int(os.environ.get("GUMBEL_MATCH_WORKSPACE_MIB", "512")) * 1024 * 1024


def sharded_gumbel_argmax(
    logits: torch.Tensor,
    keys: torch.Tensor,
    *,
    vocab_start: int,
    n_valid: int,
    group: dist.ProcessGroup | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(picks int64 [R], best float32 [R]) for this rank's vocabulary shard ``logits`` [R, V_local].

    Columns ``>= n_valid`` are vocabulary padding and never win. Every rank of ``group`` must call this with the same
    rows (TP ranks do), even when R is 0: the reduction is a collective.
    """
    rows = int(logits.shape[0])
    device = logits.device
    best = torch.full((rows,), float("-inf"), dtype=torch.float32, device=device)
    picks = torch.full((rows,), _NO_INDEX, dtype=torch.int64, device=device)
    if rows and n_valid > 0:
        ids = torch.arange(vocab_start, vocab_start + n_valid, dtype=torch.int64, device=device)
        keys = keys.to(device=device, dtype=torch.int64)
        step = max(1, _WORKSPACE_BYTES // (n_valid * 8 * 6))
        with torch.no_grad():
            for start in range(0, rows, step):
                end = min(rows, start + step)
                x = logits[start:end, :n_valid].detach().float()
                if temperature != 1.0:
                    x = x / float(temperature)
                scores = x + gumbel_noise(keys[start:end], ids)
                chunk_best, chunk_arg = scores.max(dim=-1)  # first index on a tie
                best[start:end] = chunk_best
                picks[start:end] = chunk_arg + vocab_start
    if group is not None and dist.get_world_size(group) > 1:
        global_best = best.clone()
        dist.all_reduce(global_best, op=dist.ReduceOp.MAX, group=group)
        candidate = torch.where(best == global_best, picks, torch.full_like(picks, _NO_INDEX))
        dist.all_reduce(candidate, op=dist.ReduceOp.MIN, group=group)
        return candidate, global_best
    return picks, best


@functools.lru_cache(maxsize=4)
def true_vocab_size(hf_checkpoint: str) -> int:
    """The vocabulary the miners' vLLM draws over: the HF config's lm_head rows (``vocab_size``)."""
    with open(os.path.join(hf_checkpoint, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    text = config.get("text_config", config)
    return int(text.get("unpadded_vocab_size") or text["vocab_size"])


def gumbel_match_loss_function(args, batch, logits, sum_of_sample_mean):
    """Miles Tinker loss: ``importance_sampling`` + per-datum ``vstar`` / ``match`` (see the module doc)."""
    from miles.backends.training_utils.loss.hub import tinker_losses as tl
    from miles.backends.training_utils.loss.hub.logit_processors import get_responses
    from miles.backends.training_utils.parallel import get_parallel_state

    loss, log = tl.importance_sampling_loss_function(args, batch, logits, sum_of_sample_mean)
    per_datum = log.get("per_datum") or []
    if not per_datum:
        return loss, log

    config = batch.get("loss_fn_config") or {}
    temperature = float(config.get("temperature", 1.0))
    parallel = get_parallel_state()
    tp_group = parallel.tp.group if parallel.tp.size > 1 else None
    width = int(logits.shape[-1])
    vocab_start = (parallel.tp.rank * width) if tp_group is not None else 0
    n_valid = min(max(true_vocab_size(args.hf_checkpoint) - vocab_start, 0), width)

    label_tokens = [
        torch.cat([tokens[: len(tokens) - len(targets)], tl._as_tensor_like(targets, tokens)])
        for tokens, targets in zip(batch["unconcat_tokens"], batch["target_tokens"], strict=True)
    ]
    local_keys = tl._local_response_values(args, batch, batch["draw_keys"])
    local_vstar = []
    for (logits_chunk, _), keys in zip(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=label_tokens,
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens", None),
        ),
        local_keys,
        strict=True,
    ):
        key_tensor = torch.as_tensor(list(keys), dtype=torch.int64, device=logits.device)
        picks, _ = sharded_gumbel_argmax(
            logits_chunk, key_tensor, vocab_start=vocab_start, n_valid=n_valid, group=tp_group,
            temperature=temperature,
        )
        local_vstar.append(picks)

    response_lengths = batch["response_lengths"]
    if parallel.cp.size > 1:
        positions = tl._local_response_values(
            args, batch, [torch.arange(n, device=logits.device) for n in response_lengths]
        )
        full = [
            torch.zeros(n, dtype=torch.int64, device=logits.device).index_copy_(0, pos, v)
            for v, pos, n in zip(local_vstar, positions, response_lengths, strict=True)
        ]
        flat = torch.cat(full) if full else torch.zeros(0, dtype=torch.int64, device=logits.device)
        dist.all_reduce(flat, group=parallel.cp.group)
        local_vstar = list(flat.split(list(response_lengths)))

    indices = [int(i) for i in batch["sample_indices"]]
    vstar_by_index = {i: v.cpu() for i, v in zip(indices, local_vstar, strict=True)}
    targets_by_index = dict(zip(indices, batch["target_tokens"], strict=True))
    for output in per_datum:
        index = int(output["sample_index"])
        vstar = vstar_by_index[index]
        targets = torch.as_tensor(list(targets_by_index[index]), dtype=torch.int64)
        output["vstar"] = vstar
        output["match"] = (vstar == targets).float()
    return loss, log
