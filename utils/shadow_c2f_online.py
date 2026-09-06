"""Frozen online Wan→FOCUS conditioner used while only C2F is trainable."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FOCUS_ROOT = ROOT / "external" / "FOCUS"
FOCUS_OPS = FOCUS_ROOT / "focus" / "modeling" / "pixel_decoder" / "ops"
for path in (FOCUS_ROOT, FOCUS_OPS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _pil_from_tensor(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach().float().clamp(0, 1).mul(255).round().byte()
        .permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(array, mode="RGB")


class FrozenOnlineShadowConditioner:
    """Run frozen baseline and FOCUS forward passes in the C2F process."""

    def __init__(
        self,
        *,
        device: torch.device,
        baseline_checkpoint: Path,
        wan_weights: Path,
        focus_config: Path,
        focus_checkpoint: Path,
        steps: int = 50,
        cfg_scale: float = 2.0,
    ) -> None:
        from scripts import infer_manifest as baseline

        self.device = device
        self.baseline = baseline
        baseline.ensure_runtime_imports(include_model=True)
        self.args = argparse.Namespace(
            weights_dir=str(wan_weights), checkpoint=str(baseline_checkpoint),
            device=str(device), token_dim=0, tokenlight_max_lights=2,
            fourier_features=512, fourier_sigma=5.0, height=480, width=480,
            num_frames=1, num_inference_steps=int(steps), cfg_scale=float(cfg_scale),
            seed=0, prompt="photorealistic object relighting, preserve geometry and materials",
            tokenlight_mask_tokens=False,
        )
        self.pipe, self.light_encoder, self.type_embedding = baseline.setup_pipeline(self.args)
        for module in (self.pipe.dit, self.pipe.vae, self.light_encoder, self.type_embedding):
            if module is not None and hasattr(module, "eval"):
                module.eval().requires_grad_(False)

        from detectron2.config import get_cfg
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.utils import comm as d2_comm
        from focus import add_cliprefiner_config, add_dinov2_config, add_focus_config

        # torchrun initializes the global process group, while FOCUS's CLIP
        # refiner also expects Detectron2's node-local process group.
        if torch.distributed.is_initialized() and d2_comm._LOCAL_PROCESS_GROUP is None:
            d2_comm.create_local_process_group(torch.distributed.get_world_size())

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_focus_config(cfg)
        add_cliprefiner_config(cfg)
        add_dinov2_config(cfg)
        cfg.merge_from_file(str(focus_config))
        cfg.MODEL.WEIGHTS = str(focus_checkpoint)
        cfg.MODEL.DEVICE = str(device)
        cfg.freeze()
        self.focus = DefaultPredictor(cfg)
        self.focus.model.eval().requires_grad_(False)

    def _focus_mask(self, image: Image.Image) -> torch.Tensor:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        prediction = self.focus(rgb[:, :, ::-1].copy())
        instances = prediction.get("instances")
        if instances is None or not hasattr(instances, "pred_masks"):
            raise RuntimeError("FOCUS returned no pred_masks")
        mask = instances.pred_masks
        if mask.ndim == 3:
            mask = mask[0]
        if mask.ndim != 2:
            raise RuntimeError(f"Unexpected FOCUS mask shape: {tuple(mask.shape)}")
        return mask.detach().to(device=self.device, dtype=torch.float32)

    @torch.inference_mode()
    def __call__(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if int(batch["image"].shape[0]) != 1:
            raise ValueError("Online frozen stack currently requires per-GPU batch size 1")
        source = _pil_from_tensor(batch["image"][0])
        self.args.prompt = batch["prompt"][0] or self.args.prompt
        attrs = self.baseline.parse_attrs_json(batch["attrs_json"][0])
        target = self.baseline.generate(
            self.pipe, self.light_encoder, self.type_embedding,
            attrs, source, None, self.args, extra_masks=None,
        )[0].convert("RGB")
        source_prob = self._focus_mask(source)
        target_prob = self._focus_mask(target)
        delta = (target_prob - source_prob).clamp_min(0.0)
        features = torch.stack((target_prob, source_prob, delta), dim=0).unsqueeze(0)
        return features, delta[None, None]


__all__ = ["FrozenOnlineShadowConditioner"]
