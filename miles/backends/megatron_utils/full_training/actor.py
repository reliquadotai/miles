"""Single-model Tinker commands backed by native full-parameter Megatron training.

Slot zero is the trainer's full model. The command signatures match the multi-LoRA actor, so the gateway's
checkpoint commits (``write_adapter_dir`` / ``commit_adapter_dir``) carry full checkpoints too: a committed dir
holds ``dist/`` (Megatron model + Adam state), or ``hf/`` (sampler export), and a full parametrization file.
"""

import json
from contextlib import ExitStack
from pathlib import Path

import torch.distributed as dist

from miles.backends.megatron_utils.actor import MegatronTrainRayActor
from miles.backends.megatron_utils.checkpoint import _load_checkpoint_hf
from miles.backends.megatron_utils.full_training import checkpoint
from miles.backends.megatron_utils.full_training.optimizer import GradientAccumulator
from miles.backends.megatron_utils.hf_export import save_hf_model
from miles.backends.megatron_utils.lora.model import run_forward_backward
from miles.backends.megatron_utils.optimizer_state_reset import reset_optimizer_states
from miles.backends.training_utils.checkpoint.adapter.commit import UncommittedAdapterDir, write_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.parametrization import PARAMETRIZATION_FILENAME
from miles.backends.training_utils.data.rollout import get_rollout_data
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.object_store import StoreObjectRef
from miles.utils.tracking_utils.structured_log import with_logs

STATE_SUBDIR = "dist"
HF_SUBDIR = "hf"


def full_parametrization(fields: dict) -> dict:
    """What a committed full checkpoint records; the gateway reads it back (``FullTrainingBackend.resolve_adapter``)."""
    return {
        "format_version": 1,
        "is_lora": False,
        "base_model": fields["base_model"],
        "lora_rank": None,
        "lora_alpha": None,
        "target_modules": [],
        "experts_shared_outer_loras": False,
    }


def _is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


class FullTrainingRayActor(MegatronTrainRayActor):
    def _init_training_state(self) -> None:
        self._init_weight_updater_and_publisher(update_weights=False, publish_snapshots=True)
        self.accumulator = GradientAccumulator(self.optimizer)
        self.model_is_active = False
        self.model_is_pristine = True

    @with_logs
    def load_slot(self, slot: int, rank, alpha, source: dict | None = None) -> None:
        """``source``: ``path`` (a committed full checkpoint dir) and ``load_optimizer``; None starts from the HF base."""
        assert slot == 0, "full training has one model"
        self._heartbeat.bump()
        self.accumulator.clear()
        if source is not None:
            checkpoint.load(
                self.args, self.model, self.optimizer, str(Path(source["path"]) / STATE_SUBDIR),
                load_optimizer=source["load_optimizer"],
            )
        elif not self.model_is_pristine:
            _load_checkpoint_hf(self.model, self.optimizer, self.args, self.args.hf_checkpoint)
            reset_optimizer_states(self.optimizer)
        self._zero_grads()
        self.model_is_active = True
        self.model_is_pristine = False

    @with_logs
    def unload_slot(self, slot: int) -> None:
        assert slot == 0
        self.accumulator.clear()
        self._zero_grads()
        self.model_is_active = False

    def _zero_grads(self) -> None:
        self.optimizer.zero_grad()
        for chunk in self.model:
            chunk.zero_grad_buffer()

    @with_logs
    def forward_backward(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        return self._execute_batch(batch_id, rollout_data_ref, forward_only=False)

    @with_logs
    def forward_only(self, batch_id: int, rollout_data_ref: StoreObjectRef) -> dict:
        return self._execute_batch(batch_id, rollout_data_ref, forward_only=True)

    def _execute_batch(self, batch_id: int, rollout_data_ref: StoreObjectRef, *, forward_only: bool) -> dict:
        assert self.model_is_active
        self._heartbeat.bump()
        if not forward_only:
            self._zero_grads()
        with ExitStack() as stack:
            rollout_data, store_get_result = get_rollout_data(
                args=self.args,
                rollout_data_ref=rollout_data_ref,
                output_store_read=self.output_store_read,
                train_parallel_config=get_parallel_state().train_parallel_config(supports_precomputed_schedule=True),
                token_space_processor=self.token_space_processor,
            )
            stack.enter_context(store_get_result)
            result = run_forward_backward(
                self.args, batch_id, self.model, None, rollout_data, forward_only=forward_only
            )
        if not forward_only:
            self.accumulator.add()
        self.prof.step(rollout_id=batch_id)
        return result

    @with_logs
    def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        assert self.model_is_active and set(adam_params_by_slot) == {0}
        self._heartbeat.bump()
        result = self.accumulator.step(adam_params_by_slot[0])
        self._zero_grads()
        return {0: result}

    def _write_parametrization(self, path: Path, fields: dict) -> None:
        if _is_rank0():
            (path / PARAMETRIZATION_FILENAME).write_text(json.dumps(full_parametrization(fields)))

    @with_logs
    def save_slot(
        self, slot: int, root: str, name: str, tenant_digest: str, parametrization_fields: dict
    ) -> UncommittedAdapterDir:
        """Model + Adam state under ``root``, uncommitted; the gateway commits it."""
        assert self.model_is_active and slot == 0
        self._heartbeat.bump()
        if self.accumulator.num_batches:
            raise RuntimeError("save_state requires optim_step first; pending gradients are not checkpointed")
        with write_adapter_dir(Path(root), name, tenant_digest=tenant_digest) as written:
            checkpoint.save(self.args, self.model, self.optimizer, str(written.path / STATE_SUBDIR))
            self._write_parametrization(written.path, parametrization_fields)
        return written

    @with_logs
    def export_slot(
        self, slot: int, root: str, name: str, tenant_digest: str, parametrization_fields: dict
    ) -> UncommittedAdapterDir:
        """HF weights for samplers (and miners) under ``root``, uncommitted; the gateway commits it."""
        assert self.model_is_active and slot == 0
        self._heartbeat.bump()
        with write_adapter_dir(Path(root), name, tenant_digest=tenant_digest) as written:
            save_hf_model(
                self.args,
                0,
                self.model,
                publisher=self.snapshot_publisher,
                adapter_slots=None,
                path=written.path / HF_SUBDIR,
                raise_on_error=True,
            )
            self._write_parametrization(written.path, parametrization_fields)
        return written
