"""Reliquary's Tinker gateway: one full-parameter model trained through the Tinker API with the ``reliquary_gate``
loss, training only (no SGLang: miners sample the published checkpoints). Started by Reliquary's async trainer
(``RELIQUARY_TRAINER_BACKEND=miles``):

    python scripts/run_reliquary_tinker.py serve --hf-checkpoint DIR --model-type qwen3.5-9B --tp 2 \\
        --actor-num-gpus 8 --num-gpus-per-node 8 --output-dir OUT --extra-args "--tinker-base-model NAME"

``--mode lora`` / ``--inference`` keep the multi-LoRA and SGLang variants."""

from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)

    hf_checkpoint: str | None = None
    model_type: str = "qwen3-8B"
    model_dir: str = "/root/models"
    save_dir: str | None = None
    megatron_path: str = "/root/Megatron-LM"
    # bridge: Megatron-Bridge LoRA; raw: miles-native LoRA, dense models only (not this default MoE)
    megatron_to_hf_mode: Literal["bridge", "raw"] = "bridge"

    num_gpus_per_node: int = 8
    actor_num_gpus: int = 8
    rollout_num_gpus: int = 0
    tp: int = 2
    ep: int = 1

    # LoRA slot pool; per-client rank comes from the SDK, capped by lora_rank.
    lora_rank: int = 32
    lora_alpha: int = 64
    n_adapters: int = 4
    target_modules: str = "attn,mlp,unembed"

    tinker_port: int = 10613
    rollout_num_gpus_per_engine: int = 1
    sglang_mem_fraction_static: float = 0.7

    extra_args: str = ""
    # "full" (one full-parameter model) or "lora" (multi-LoRA slots)
    mode: str = "full"
    # False: training-only gateway (--debug-train-only), no SGLang, no rollout GPUs; miners sample the exports
    inference: bool = False

    def __post_init__(self):
        if self.save_dir is None:
            self.save_dir = f"{self.output_dir}/checkpoints"
        if self.hf_checkpoint is None:
            raise ValueError("--hf-checkpoint is required")


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs):
    """Serve the Tinker gateway (idles until clients connect)."""
    print(
        f"[run] tinker gateway: {args.actor_num_gpus} train + {args.rollout_num_gpus} rollout GPUs, "
        f"{args.n_adapters} adapter slots, port {args.tinker_port}"
    )

    # raw mode starts fresh runs from --ref-load
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.hf_checkpoint} "
        f"--megatron-to-hf-mode {args.megatron_to_hf_mode} "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout 0.0 "
        "--no-gradient-accumulation-fusion "
        f"--multi-lora-n-adapters {args.n_adapters} "
        f'--target-modules "{args.target_modules}"'
    )
    if args.mode == "full":
        lora_args = "--tinker-full-training --use-distributed-optimizer --no-gradient-accumulation-fusion"

    tinker_args = f"--tinker-server-port {args.tinker_port} " f"--tinker-checkpoint-root {args.save_dir}/{args.run_id}"

    # initial config only; AdamParams come per optim_step request
    optimizer_args = "--optimizer adam --lr 1e-4 "

    perf_args = (
        f"--tensor-model-parallel-size {args.tp} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--expert-model-parallel-size {args.ep} --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
        # Tinker targets cover full sequences; bound the FP32 cross-entropy temporaries
        "--log-probs-chunk-size 1024 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-lora-backend triton "
    )

    topology_args = (
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.actor_num_gpus} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
    )
    if not args.inference:
        sglang_args = ""
        topology_args = f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.actor_num_gpus} --debug-train-only "

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
    )

    train_args = (
        f"{ckpt_args} {lora_args} {tinker_args} {optimizer_args} "
        f"{perf_args} {sglang_args} {topology_args} {misc_args} {args.extra_args} "
    )

    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.model_type,
        train_script="serve_tinker.py",
        megatron_path=args.megatron_path,
        job_lifetime="launcher",
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
