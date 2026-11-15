"""Full model and native Adam state in Megatron's distributed checkpoint format."""

import os

import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.utils import unwrap_model
from megatron.training.checkpointing import _build_sharded_state_dict_metadata

from miles.backends.megatron_utils.optimizer_state_reset import reset_optimizer_states
from miles.backends.training_utils.parallel import get_parallel_state


def _state(args, model, optimizer, *, is_loading: bool) -> dict:
    metadata = _build_sharded_state_dict_metadata(args, dp_cp_group=get_parallel_state().intra_dp_cp.group)
    chunks = unwrap_model(model)
    state = {
        "model" if len(chunks) == 1 else f"model{index}": chunk.sharded_state_dict(metadata=metadata)
        for index, chunk in enumerate(chunks)
    }
    if optimizer is not None:
        # Adam normally allocates moments on its first step. Tinker also allows
        # saving a freshly created model, before any optimizer update.
        leaves = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
        for leaf in leaves:
            if not leaf.is_stub_optimizer:
                leaf.init_state_fn(leaf.optimizer, leaf.config)
        state["optimizer"] = optimizer.sharded_state_dict(state, is_loading=is_loading, metadata=metadata)
    return state


def save(args, model, optimizer, path: str) -> None:
    """Into ``path`` (created here), inside a dir the gateway commits atomically afterwards."""
    state = _state(args, model, optimizer, is_loading=False)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(path, exist_ok=False)
    if dist.is_initialized():
        dist.barrier()
    dist_checkpointing.save(state, str(path))


def load(args, model, optimizer, path: str, *, load_optimizer: bool) -> None:
    state = _state(args, model, optimizer if load_optimizer else None, is_loading=True)
    loaded = dist_checkpointing.load(state, path)
    chunks = unwrap_model(model)
    for index, chunk in enumerate(chunks):
        chunk.load_state_dict(loaded["model" if len(chunks) == 1 else f"model{index}"])
    if load_optimizer:
        optimizer.load_state_dict(loaded["optimizer"])
    else:
        reset_optimizer_states(optimizer)
        optimizer.reload_model_params()
