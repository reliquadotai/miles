"""Accumulate Tinker commands in Megatron's own DDP gradient buffer, reduced once per optimizer step.

Every command's backward adds into the persistent ``main_grad`` buffer; the commands run without the grad
reduction (``run_forward_backward(defer_grad_sync=True)``), so nothing is reduced twice and no copy of the
gradients is kept. ``step`` runs the one reduction (``finalize``), then Adam on the reduced shards.
"""

import math
from collections.abc import Callable

import torch


class GradientAccumulator:
    def __init__(self, optimizer, finalize: Callable[[], None]) -> None:
        self.optimizer = optimizer
        self.finalize = finalize
        self.num_batches = 0
        self.poisoned = False

    def add(self) -> None:
        self.num_batches += 1

    def poison(self) -> None:
        """A command failed after part of its backward reached the buffer: the step cannot be trusted."""
        self.poisoned = True

    @torch.no_grad()
    def step(self, adam: dict) -> dict:
        if self.poisoned:
            self.clear()
            return {"error": "a forward_backward failed since the last optim_step; its gradients were dropped"}
        if not self.num_batches:
            return {"error": "optim_step requires accumulated gradients"}
        for group in self.optimizer.param_groups:
            group.update(
                lr=adam["learning_rate"],
                betas=(adam["beta1"], adam["beta2"]),
                eps=adam["eps"],
                weight_decay=adam["weight_decay"],
            )
        self.finalize()
        self.optimizer.prepare_grads()
        grad_norm = float(self.optimizer.get_grad_norm())
        if not math.isfinite(grad_norm):
            self.clear()
            return {"skipped_nonfinite": 1}
        if adam["grad_clip_norm"] > 0:
            coefficient = min(1.0, adam["grad_clip_norm"] / (grad_norm + 1e-6))
            for param in self.optimizer.get_parameters():
                if param.grad is not None:
                    param.grad.mul_(coefficient)
        successful = self.optimizer.step_with_ready_grads()
        self.clear()
        return {"grad_norm": grad_norm} if successful else {"skipped_nonfinite": 1}

    def clear(self) -> None:
        """Zero the optimizer's grads; the caller also zeroes the DDP buffers (``zero_grad_buffer``)."""
        self.optimizer.zero_grad()
        self.num_batches = 0
        self.poisoned = False
