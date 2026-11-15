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


# output rows upcast at a time: bounds the fp32 weight temporaries (16384 x 4096 x 4 B = 256 MiB)
_HEAD_CHUNK = int(os.environ.get("FP32_HEAD_CHUNK_ROWS", "16384"))


class _Fp32Head(torch.autograd.Function):
    """``fp32(x) @ fp32(w).T`` over row chunks of ``w``: no fp32 copy of the weight is kept for backward, and the
    weight gradient is computed chunk by chunk straight into the weight's dtype."""

    @staticmethod
    def forward(ctx, x, w, chunk):
        x32 = x.float()
        out = x32.new_empty(*x.shape[:-1], w.shape[0])
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for start in range(0, w.shape[0], chunk):
                end = min(w.shape[0], start + chunk)
                out[..., start:end] = x32 @ w[start:end].float().t()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        ctx.save_for_backward(x, w)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        need_x, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        g = grad.reshape(-1, w.shape[0])
        x32 = x.reshape(-1, x.shape[-1]).float()
        grad_x = torch.zeros_like(x32) if need_x else None
        grad_w = torch.empty_like(w) if need_w else None
        for start in range(0, w.shape[0], ctx.chunk):
            end = min(w.shape[0], start + ctx.chunk)
            g_chunk = g[:, start:end]
            if need_x:
                grad_x.addmm_(g_chunk, w[start:end].float())
            if need_w:
                grad_w[start:end] = (g_chunk.t() @ x32).to(w.dtype)
        if need_x:
            grad_x = grad_x.view(x.shape).to(x.dtype)
        return grad_x, grad_w, None


def fp32_head(x: torch.Tensor, w: torch.Tensor, *, chunk: int = _HEAD_CHUNK) -> torch.Tensor:
    """The miners' ``head_dtype=float32`` lm_head: ``fp32(x) @ fp32(w).T`` in exact fp32 (TF32 off)."""
    return _Fp32Head.apply(x, w, chunk)


def _output_linears(model) -> list:
    """The vocab-parallel output projection(s) of a Megatron model (list of chunks), innermost module only."""
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    chunks = model if isinstance(model, (list, tuple)) else [model]
    found = []
    for chunk in chunks:
        for name, module in chunk.named_modules():
            if "output_layer" in name.split(".")[-2:] or name.endswith("output_layer"):
                if isinstance(module, ColumnParallelLinear):
                    found.append(module)
    return found


class fp32_output_layer:
    """While active, the output projection computes ``fp32(h) @ fp32(W)`` in exact fp32 (TF32 off): the miners'
    ``head_dtype=float32`` head, so v* is decided on the logits the draw saw, not on bf16-rounded ones."""

    def __init__(self, model, enabled: bool):
        self.model, self.enabled, self.saved = model, enabled, []

    def __enter__(self):
        if not self.enabled:
            return self
        from megatron.core.tensor_parallel.mappings import (
            copy_to_tensor_model_parallel_region,
            gather_from_sequence_parallel_region,
        )

        linears = _output_linears(self.model)
        if not linears:
            raise RuntimeError("gumbel_match: no output_layer ColumnParallelLinear found for the fp32 head")
        for linear in linears:
            original = linear.forward

            def forward(input_, weight=None, runtime_gather_output=None, _linear=linear, **kwargs):
                if kwargs:
                    raise RuntimeError(f"gumbel_match fp32 head: unexpected output_layer args {sorted(kwargs)}")
                w = _linear.weight if weight is None else weight
                group = getattr(_linear, "tp_group", None)
                if _linear.sequence_parallel:
                    x = gather_from_sequence_parallel_region(input_, group=group)
                else:
                    x = copy_to_tensor_model_parallel_region(input_, group=group)
                out = fp32_head(x, w)
                gather = _linear.gather_output if runtime_gather_output is None else runtime_gather_output
                if gather:
                    raise RuntimeError("gumbel_match fp32 head: gathered output is not supported")
                return out, None

            self.saved.append((linear, original))
            linear.forward = forward
        return self

    def __exit__(self, *exc):
        for linear, original in self.saved:
            linear.forward = original
        self.saved = []
        return False
