from miles.utils.lora.hf_lora_targets import LORA_TARGET_GROUPS, parse_lora_targets


def configure_tinker_args(args):
    assert args.train_backend == "megatron", "Tinker requires the Megatron backend"
    assert args.exclude_modules is None, "Tinker selects complete training groups; --exclude-modules is not supported"
    if getattr(args, "tinker_full_training", False):
        # trainer flags live on args or, after the args refactor, on args.backend
        def flag(name):
            backend = getattr(args, "backend", None)
            if hasattr(args, name):
                return getattr(args, name)
            return getattr(backend, name, None) if backend is not None else None

        assert (
            not args.multi_lora_n_adapters and not args.lora_rank and args.lora_adapter_path is None
        ), "full training cannot enable LoRA"
        assert flag("bf16") and not flag("fp16"), "Tinker full training currently requires BF16"
        assert flag("loss_scale") in (None, 1.0), "Tinker full training requires unit loss scaling"
        assert flag("optimizer") == "adam", "Tinker optim_step supplies Adam parameters"
        assert not flag("calculate_per_token_loss"), "Tinker loss inputs already contain client normalization"
        assert (
            flag("use_dynamic_batch_size") or flag("micro_batch_size") == 1
        ), "full training requires --use-dynamic-batch-size or --micro-batch-size 1 for arbitrary SDK batches"
        assert flag("context_parallel_size") in (None, 1), "Tinker full training does not yet support context parallelism"
        assert not flag("offload_train") and not flag("colocate"), "Tinker requires resident trainer and inference GPUs"
        assert not flag("overlap_param_gather"), "Tinker full training does not yet support overlapping parameter gathers"
        assert not flag("use_precision_aware_optimizer"), "Tinker full training requires ordinary BF16 Adam"
        for name in (
            "indep_dp",
            "debug_disable_optimizer",
            "debug_rollout_only",
            "enable_mtp_training",
            "optimizer_cpu_offload",
            "stream_optimizer_state_to_disk",
            "rematerialize_param_from_master_weight",
        ):
            assert not flag(name), f"--{name.replace('_', '-')} is not supported with Tinker full training"
        args.tinker_lora_groups = []
        return
    groups = parse_lora_targets(args.target_modules)
    if groups is None:
        groups = list(LORA_TARGET_GROUPS)
    assert set(groups) <= set(LORA_TARGET_GROUPS), "Tinker --target-modules accepts only attn,mlp,unembed groups"
    args.tinker_lora_groups = groups
    args.target_modules = groups
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True


def validate_tinker_args(args):
    """Checks on the final arguments; preprocessing runs before --custom-config-path is merged."""
    # --save-interval and the other save-loop flags already require --save in miles_validate_args
    for flag, value in (("--save", args.save), ("--save-hf", args.save_hf)):
        assert value is None, f"Tinker does not support {flag}; checkpoints are saved through the Tinker API"
    assert args.tinker_checkpoint_root, "Tinker requires --tinker-checkpoint-root"
