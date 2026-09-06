from __future__ import annotations

"""Shared safe launcher for the new physics-conditioned trainers.

This module is intentionally separate from the existing TokenLight launchers.
It keeps FP32 trainable parameters for plain single-GPU AdamW, lets DeepSpeed
own its ZeRO-3 precision policy, clips only outside DeepSpeed, and uses a true
constant learning-rate scheduler from the first optimizer step.
"""

import torch

from model import train_tokenlight as base
from model import train_tokenlight_decoder_safe as safe_legacy


def is_deepspeed(accelerator) -> bool:
    return safe_legacy._is_deepspeed(accelerator)


def select_trainable_dtype(args, accelerator) -> torch.dtype:
    return safe_legacy._select_trainable_dtype(args, accelerator)


def launch_physics_training_task(
    accelerator,
    dataset,
    model,
    model_logger,
    *,
    args,
) -> None:
    """Train one latent-only physics task with stable optimizer semantics."""

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if float(getattr(args, "tokenlight_rgb_decoder_loss_weight", 0.0)) != 0.0:
        raise ValueError("physics runtime is latent-only; RGB decoder loss must be zero")

    trainable_dtype = select_trainable_dtype(args, accelerator)
    before = base._coerce_trainable_parameter_dtype(model, trainable_dtype)
    after = base._trainable_dtype_counts(model)
    if accelerator.is_main_process:
        print(f"Trainable dtype: target={trainable_dtype}, before={before}, after={after}")

    optimizer_class = base.get_optimizer_class(args.customized_optimizer)
    trainable_parameters = base._trainable_parameters(model)
    optimizer = optimizer_class(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # torch ConstantLR defaults to factor=1/3 for five steps.  A no-op LambdaLR
    # expresses the intended constant LR without that startup reduction.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    task_batch = base._parse_balanced_task_batch(
        getattr(args, "balanced_task_batch", None)
    )
    dataloader_kwargs = base._dataloader_runtime_kwargs(
        args,
        dataset,
        num_workers=args.dataset_num_workers,
        accelerator=accelerator,
    )
    if task_batch is None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=base._collate_tokenlight_batch,
            drop_last=batch_size > 1,
            **dataloader_kwargs,
        )
    else:
        if sum(task_batch.values()) != batch_size:
            raise ValueError(
                f"balanced_task_batch sums to {sum(task_batch.values())}, "
                f"batch={batch_size}"
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=base.BalancedTaskBatchSampler(
                dataset.data,
                task_batch,
                seed=int(getattr(args, "balanced_batch_seed", 0)),
            ),
            collate_fn=base._collate_tokenlight_batch,
            **dataloader_kwargs,
        )

    base._configure_deepspeed_batch_size(accelerator, args, batch_size)
    enable_model_cpu_offload = bool(
        getattr(args, "enable_model_cpu_offload", False)
    )
    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(
            optimizer, dataloader, scheduler
        )
        model.pipe.device = accelerator.device
        offload_manager = base.OffloadTrainingManager(
            model,
            accelerator.device,
            getattr(args, "enable_optimizer_cpu_offload", False),
            getattr(args, "cpu_offload_split_threshold", None),
        )
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
        offload_manager = None

    base.save_training_runtime_snapshot(args, accelerator, model)
    tb_metrics = base.TokenLightTensorBoardMetrics(
        args.output_path,
        enabled=args.enable_tensorboard_log,
    )
    base.initialize_deepspeed_gradient_checkpointing(accelerator)
    optimizer_step = int(getattr(model_logger, "num_steps", 0))
    try:
        for epoch_id in range(
            int(args.start_epoch), int(args.start_epoch) + int(args.num_epochs)
        ):
            iterator = base.tqdm(
                dataloader, disable=not accelerator.is_local_main_process
            )
            for data in iterator:
                unwrapped = accelerator.unwrap_model(model)
                # Existing training modules expose this field for loss ramps;
                # keeping it updated is harmless for latent-only objectives.
                unwrapped.decoder_optimizer_step = optimizer_step
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        loss = model({}, inputs=data) if dataset.load_from_cache else model(data)
                    accelerator.backward(loss)
                    if offload_manager is not None:
                        offload_manager.after_backward()
                    if (
                        accelerator.sync_gradients
                        and not is_deepspeed(accelerator)
                        and float(args.max_grad_norm) > 0
                    ):
                        accelerator.clip_grad_norm_(
                            trainable_parameters, float(args.max_grad_norm)
                        )
                    optimizer.step()
                    scheduler.step()
                    if accelerator.sync_gradients:
                        metrics = base._collect_train_metrics(
                            accelerator, model, optimizer, loss
                        )
                        optimizer_step += 1
                        base._call_compatible_method(
                            model_logger,
                            "on_step_end",
                            accelerator,
                            model,
                            args.save_steps,
                            loss=loss,
                            epoch=epoch_id,
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )
                        tb_metrics.log(accelerator, optimizer_step, metrics)
                        if metrics and "train/loss" in metrics:
                            iterator.set_postfix(
                                loss=f"{metrics['train/loss']:.4f}",
                                step=optimizer_step,
                            )
                    optimizer.zero_grad(set_to_none=True)
            if args.save_steps is None:
                base._call_compatible_method(
                    model_logger,
                    "on_epoch_end",
                    accelerator,
                    model,
                    epoch_id,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
        base._call_compatible_method(
            model_logger,
            "on_training_end",
            accelerator,
            model,
            args.save_steps,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    finally:
        tb_metrics.close(accelerator)


__all__ = [
    "is_deepspeed",
    "launch_physics_training_task",
    "select_trainable_dtype",
]
