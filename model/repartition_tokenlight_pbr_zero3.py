from __future__ import annotations

import inspect
import json
import re
import shutil
import sys
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import accelerate
import torch

from model.train_tokenlight import (
    _coerce_trainable_parameter_dtype,
    _make_model_logger,
    _maybe_load_full_training_state,
    _preferred_trainable_dtype,
    _save_full_training_state,
    _trainable_dtype_counts,
    _trainable_parameters,
    save_training_config_snapshot,
    save_training_runtime_snapshot,
)
from model.train_tokenlight_pbr import (
    TokenLightPBRWanTrainingModule,
    _coerce_floating_module_dtype,
    _configure_deepspeed_batch_size,
    _floating_module_dtype_counts,
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
    parse_tokenlight_pbr_args,
)


def _step_from_checkpoint_path(path: str | None) -> int | None:
    if path in (None, "", "None", "none", "null"):
        return None
    match = re.search(r"(?:^|/)full-step-(\d+)(?:/)?$", str(path).rstrip("/"))
    if match:
        return int(match.group(1))
    return None


def _checkpoint_label(args, model_logger) -> str:
    label = getattr(args, "repartition_label", None)
    if label not in (None, "", "None", "none", "null"):
        return str(label)
    step = int(getattr(model_logger, "num_steps", 0) or 0)
    if step <= 0:
        step = _step_from_checkpoint_path(getattr(args, "resume_from_checkpoint", None)) or 0
    if step <= 0:
        raise ValueError(
            "Could not infer repartition checkpoint step. "
            "Pass --repartition_label step-N or use a full-step-N resume checkpoint."
        )
    return f"step-{step}"


def _checkpoint_tag(checkpoint_dir: Path) -> str:
    latest = checkpoint_dir / "latest"
    if latest.exists():
        value = latest.read_text(encoding="utf-8").strip()
        if value:
            return value
    return "pytorch_model"


def _looks_like_universal_checkpoint(checkpoint_dir: Path) -> bool:
    latest = checkpoint_dir / "latest_universal"
    if latest.exists():
        tag = latest.read_text(encoding="utf-8").strip()
        if tag and (checkpoint_dir / tag / "zero" / "optimizer_state.pt").exists():
            return True
    tag = _checkpoint_tag(checkpoint_dir)
    return (checkpoint_dir / tag / "zero" / "optimizer_state.pt").exists()


def _copy_accelerate_sidecar_files(source_dir: Path, output_dir: Path, *, source_checkpoint: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("scheduler.bin", "scaler.pt", "random_states_*.pkl", "custom_checkpoint_*.pkl"):
        for path in source_dir.glob(pattern):
            if path.is_file():
                shutil.copy2(path, output_dir / path.name)
    metadata_path = source_dir / "checkpoint_metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    extra = metadata.get("extra") if isinstance(metadata.get("extra"), dict) else {}
    extra = dict(extra)
    extra["universal_converted_from"] = source_checkpoint
    metadata = {
        "label": metadata.get("label", source_dir.name),
        "format": "accelerate.save_state+deepspeed.universal",
        "extra": extra,
    }
    with (output_dir / "checkpoint_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def _prepare_universal_checkpoint(args, accelerator) -> Path:
    source_dir = Path(str(args.resume_from_checkpoint))
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source checkpoint directory: {source_dir}")
    if _looks_like_universal_checkpoint(source_dir):
        return source_dir

    output_dir = (
        Path(str(args.universal_checkpoint_dir))
        if args.universal_checkpoint_dir not in (None, "", "None", "none", "null")
        else source_dir.with_name(f"{source_dir.name}_universal")
    )
    tag = _checkpoint_tag(source_dir)
    input_folder = source_dir / tag
    output_folder = output_dir / tag
    if not input_folder.exists():
        raise FileNotFoundError(f"Missing DeepSpeed checkpoint tag folder: {input_folder}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if output_folder.exists():
            if bool(getattr(args, "universal_overwrite", False)):
                shutil.rmtree(output_folder)
            elif _looks_like_universal_checkpoint(output_dir):
                print(f"Using existing Universal checkpoint: {output_dir}")
            else:
                raise FileExistsError(
                    f"{output_folder} already exists but does not look complete. "
                    "Pass --universal_overwrite to rebuild it."
                )
        if not _looks_like_universal_checkpoint(output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            print(
                "Converting regular DeepSpeed checkpoint to Universal checkpoint: "
                f"input={input_folder} output={output_folder}"
            )
            from deepspeed.checkpoint import ds_to_universal

            ds_to_universal.main(
                Namespace(
                    input_folder=str(input_folder),
                    output_folder=str(output_folder),
                    num_extract_workers=int(getattr(args, "universal_num_extract_workers", 1)),
                    num_merge_workers=int(getattr(args, "universal_num_merge_workers", 1)),
                    keep_temp_folder=False,
                    no_strict=False,
                    inject_missing_state=True,
                )
            )
        _copy_accelerate_sidecar_files(source_dir, output_dir, source_checkpoint=str(source_dir))
    accelerator.wait_for_everyone()
    if not _looks_like_universal_checkpoint(output_dir):
        raise RuntimeError(f"Universal checkpoint conversion did not produce a loadable checkpoint: {output_dir}")
    return output_dir


def _build_model(args, accelerator) -> TokenLightPBRWanTrainingModule:
    return TokenLightPBRWanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=getattr(args, "use_gradient_checkpointing_offload", False),
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu"
        if (args.initialize_model_on_cpu or getattr(args, "enable_model_cpu_offload", False))
        else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        tokenlight_light_tokens=args.tokenlight_light_tokens,
        tokenlight_attrs_key=args.tokenlight_attrs_key,
        tokenlight_token_dim=args.tokenlight_token_dim,
        tokenlight_fourier_features=args.tokenlight_fourier_features,
        tokenlight_fourier_sigma=args.tokenlight_fourier_sigma,
        tokenlight_max_lights=args.tokenlight_max_lights,
        tokenlight_light_dropout=args.tokenlight_light_dropout,
        tokenlight_cfg_drop_prob=args.tokenlight_cfg_drop_prob,
        tokenlight_source_tokens=args.tokenlight_source_tokens,
        tokenlight_pbr_image_key=args.tokenlight_pbr_image_key,
        tokenlight_pbr_mode_key=args.tokenlight_pbr_mode_key,
        tokenlight_pbr_aux_type=args.tokenlight_pbr_aux_type,
        tokenlight_pbr_loss_weight=args.tokenlight_pbr_loss_weight,
        tokenlight_pbr_streams=args.tokenlight_pbr_streams,
        tokenlight_pbr_stream_image_keys=args.tokenlight_pbr_stream_image_keys,
        tokenlight_pbr_stream_loss_weights=args.tokenlight_pbr_stream_loss_weights,
        tokenlight_pbr_default_mode=args.tokenlight_pbr_default_mode,
        tokenlight_pbr_conditioning_strategy=args.tokenlight_pbr_conditioning_strategy,
        tokenlight_pbr_unirelight_target_prob=args.tokenlight_pbr_unirelight_target_prob,
        tokenlight_pbr_unirelight_condition_prob=args.tokenlight_pbr_unirelight_condition_prob,
        tokenlight_pbr_unirelight_source_drop_prob=args.tokenlight_pbr_unirelight_source_drop_prob,
        tokenlight_source_drop_rgb_loss_weight=args.tokenlight_source_drop_rgb_loss_weight,
        tokenlight_source_drop_log_luminance_loss_weight=args.tokenlight_source_drop_log_luminance_loss_weight,
        tokenlight_source_drop_illum_loss_weight=args.tokenlight_source_drop_illum_loss_weight,
        tokenlight_source_drop_illum_head_path=args.tokenlight_source_drop_illum_head_path,
        tokenlight_source_drop_illum_cache_dir=args.tokenlight_source_drop_illum_cache_dir,
        tokenlight_source_drop_illum_cache_key=args.tokenlight_source_drop_illum_cache_key,
        tokenlight_source_drop_illum_cache_open_shards=args.tokenlight_source_drop_illum_cache_open_shards,
        tokenlight_source_drop_illum_latent_normalize=args.tokenlight_source_drop_illum_latent_normalize,
        tokenlight_source_drop_illum_latent_norm_eps=args.tokenlight_source_drop_illum_latent_norm_eps,
        tokenlight_log_luminance_eps=args.tokenlight_log_luminance_eps,
        tokenlight_log_luminance_decode_tiled=args.tokenlight_log_luminance_decode_tiled,
        dataset_base_path=args.dataset_base_path,
    )


def main() -> None:
    args, raw_config, merged_config = parse_tokenlight_pbr_args("zero3")
    source_checkpoint = getattr(args, "resume_from_checkpoint", None)
    if source_checkpoint in (None, "", "None", "none", "null"):
        raise ValueError("--resume_from_checkpoint is required for full-checkpoint repartitioning.")

    accelerator_kwargs = {
        "kwargs_handlers": [
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    }
    if "even_batches" in inspect.signature(accelerate.Accelerator).parameters:
        accelerator_kwargs["even_batches"] = False
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    if hasattr(accelerator, "even_batches"):
        accelerator.even_batches = False

    universal_checkpoint = _prepare_universal_checkpoint(args, accelerator)
    args.resume_from_checkpoint = str(universal_checkpoint)

    save_training_config_snapshot(args, raw_config, merged_config, accelerator)
    model = _build_model(args, accelerator)
    trainable_dtype = _preferred_trainable_dtype(model)
    before_dtype_counts = _coerce_trainable_parameter_dtype(model, trainable_dtype)
    after_dtype_counts = _trainable_dtype_counts(model)
    illum_head = getattr(model, "source_drop_illum_head", None)
    illum_head_before_dtype_counts = _coerce_floating_module_dtype(illum_head, trainable_dtype)
    illum_head_after_dtype_counts = _floating_module_dtype_counts(illum_head)
    if accelerator.is_main_process:
        print(
            "Repartition dtype setup: "
            f"target={trainable_dtype}, before={before_dtype_counts}, after={after_dtype_counts}"
        )
        if illum_head is not None:
            print(
                "Source-drop illumination head dtype setup: "
                f"target={trainable_dtype}, before={illum_head_before_dtype_counts}, "
                f"after={illum_head_after_dtype_counts}"
            )

    optimizer_class = get_optimizer_class(args.customized_optimizer)
    optimizer = optimizer_class(_trainable_parameters(model), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    _configure_deepspeed_batch_size(accelerator, args, int(args.batch_size))
    model.to(device=accelerator.device)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if hasattr(model, "load_universal_checkpoint") and not model.load_universal_checkpoint():
        raise RuntimeError(
            "This repartition path requires DeepSpeed Universal Checkpoint loading. "
            "Use configs/accelerate_zero3_load_universal.yaml or a DeepSpeed config with "
            "`checkpoint.load_universal=true`."
        )

    model_logger = _make_model_logger(args)
    _maybe_load_full_training_state(args, accelerator, model_logger, model=model)
    save_training_runtime_snapshot(args, accelerator, model)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    label = _checkpoint_label(args, model_logger)
    checkpoint_dir = Path(args.output_path) / f"full-{label}"
    if checkpoint_dir.exists():
        if not bool(getattr(args, "repartition_overwrite", False)):
            raise FileExistsError(
                f"{checkpoint_dir} already exists. Pass --repartition_overwrite to replace it."
            )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            shutil.rmtree(checkpoint_dir)
        accelerator.wait_for_everyone()

    step = int(getattr(model_logger, "num_steps", 0) or (_step_from_checkpoint_path(source_checkpoint) or 0))
    _save_full_training_state(
        accelerator,
        model,
        args.output_path,
        label,
        optimizer=optimizer,
        scheduler=scheduler,
        extra={
            "num_steps": step,
            "global_step": step,
            "repartitioned_from": str(source_checkpoint),
            "universal_checkpoint": str(universal_checkpoint),
            "target_num_processes": int(getattr(accelerator, "num_processes", 1)),
        },
    )
    if accelerator.is_main_process:
        print(
            "Repartitioned full checkpoint: "
            f"source={source_checkpoint} universal={universal_checkpoint} "
            f"output={checkpoint_dir} num_processes={accelerator.num_processes}"
        )


if __name__ == "__main__":
    main()
