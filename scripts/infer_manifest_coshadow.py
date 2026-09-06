#!/usr/bin/env python3
"""CoShadow manifest inference using GT-shadow-derived bbox tokens."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from model.infer_tokenlight import encode_image_latents  # noqa: E402
from model.tokenlight_wan_coshadow import CoShadowLayoutTokenEmbedding, model_fn_wan_video_tokenlight_coshadow  # noqa: E402
from scripts import infer_manifest as base  # noqa: E402
from scripts.infer_manifest_spatial import extract_module, parse_args as spatial_parse_args  # noqa: E402


def bbox_bins(row, args):
    if isinstance(row.get("coshadow_bbox_bins"), list):
        return torch.tensor([row["coshadow_bbox_bins"]], dtype=torch.long), bool(row.get("coshadow_bbox_valid", True))
    value = row.get("shadow_mask")
    if not value: raise KeyError("GT-bbox inference requires shadow_mask or coshadow_bbox_bins")
    path = Path(value); path = path if path.is_absolute() else Path(args.base_path) / path
    mask = np.asarray(Image.open(path).convert("L")) > 127
    ys, xs = np.where(mask)
    if not len(xs): return torch.zeros(1, 4, dtype=torch.long), False
    h, w = mask.shape; box = np.array([xs.min()/w, ys.min()/h, (xs.max()+1)/w, (ys.max()+1)/h])
    return torch.from_numpy(np.floor(np.clip(box, 0, 1) * 15 + 0.5).astype(np.int64)).unsqueeze(0), True


def setup(args):
    pipe, light, type_embedding = base.setup_pipeline(args); state = base.load_state(args.checkpoint)
    layout = CoShadowLayoutTokenEmbedding(int(pipe.dit.dim), bins=16).to(pipe.device, pipe.torch_dtype)
    layout.load_state_dict(extract_module(state, "coshadow_layout_embedding"), strict=True); layout.eval()
    return pipe, light, type_embedding, layout


@torch.no_grad()
def generate(pipe, light, type_embedding, layout, row, source, object_mask, args):
    pipe.model_fn = lambda **kw: model_fn_wan_video_tokenlight_coshadow(
        tokenlight_light_encoder=light, tokenlight_type_embedding=type_embedding,
        coshadow_layout_embedding=layout, coshadow_collect_actual_alignment=False,
        coshadow_mask_head=None, **kw)
    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=5.0)
    category = str(row.get("coshadow_category") or "object").strip() or "object"
    prompt = f"{category} casting shadow"
    pos, neg = {"prompt": prompt}, {"prompt": prompt}
    shared = {"input_image": None, "end_image": None, "input_video": None, "denoising_strength": 1.0,
              "control_video": None, "reference_image": None, "seed": args.seed, "rand_device": pipe.device,
              "height": args.height, "width": args.width, "num_frames": 1, "cfg_scale": 1, "cfg_merge": False,
              "sigma_shift": 5.0, "tiled": True, "tile_size": (30,52), "tile_stride": (15,26),
              "framewise_decoding": False}
    for unit in pipe.units: shared, pos, neg = pipe.unit_runner(unit, pipe, shared, pos, neg)
    bins, valid = bbox_bins(row, args)
    shared.update(tokenlight_attrs=[base.attrs_from_row(row, args.attrs_key)],
                  tokenlight_source_latents=encode_image_latents(pipe, source, args),
                  tokenlight_mask_latents=encode_image_latents(pipe, object_mask.convert("RGB"), args),
                  coshadow_layout_bins=bins.to(pipe.device),
                  coshadow_bbox_valid=torch.tensor([valid], device=pipe.device))
    pos["tokenlight_drop_light"] = False; neg["tokenlight_drop_light"] = True
    pipe.load_models_to_device(pipe.in_iteration_models); models={n:getattr(pipe,n) for n in pipe.in_iteration_models}
    for i, timestep in enumerate(pipe.scheduler.timesteps):
        t=timestep.unsqueeze(0).to(dtype=pipe.torch_dtype,device=pipe.device)
        pred=pipe.model_fn(**models,**shared,**pos,timestep=t)
        if isinstance(pred,tuple): pred=pred[0]
        if args.cfg_scale != 1.0:
            uncond=pipe.model_fn(**models,**shared,**neg,timestep=t); uncond=uncond[0] if isinstance(uncond,tuple) else uncond
            pred=uncond+args.cfg_scale*(pred-uncond)
        shared["latents"]=pipe.scheduler.step(pred,timestep,shared["latents"])
    pipe.load_models_to_device(["vae"]); decoded=pipe.vae.decode(shared["latents"],device=pipe.device,tiled=True,tile_size=(30,52),tile_stride=(15,26))
    return pipe.vae_output_to_video(decoded)


def main():
    args=spatial_parse_args(); args.manifest=base.resolve_repo(args.manifest).as_posix(); args.checkpoint=base.resolve_repo(args.checkpoint).as_posix()
    rows=base.load_rows(Path(args.manifest),args.limit); out=base.resolve_repo(args.output_dir); base.ensure_runtime_imports(include_model=False)
    pipe,light,type_embedding,layout=setup(args)
    for row in rows:
        pred=base.prediction_path(out,row); target=base.target_path(row,args)
        if args.skip_existing and pred.exists(): continue
        source=Image.open(base.source_path(row,args)).convert("RGB"); mask_path=Path(args.base_path)/str(row["mask"]); mask=Image.open(mask_path).convert("L")
        video=generate(pipe,light,type_embedding,layout,row,source,mask,args); pred.parent.mkdir(parents=True,exist_ok=True); video[0].save(pred)
        if args.with_gt and target and target.exists(): base.save_with_gt(pred,source,target)
    if args.eval:
        import json
        metrics=base.run_eval(rows,out,args); target=Path(args.metrics_output) if args.metrics_output else out/"metrics.json"; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(metrics,indent=2)+"\n")
    return 0


if __name__ == "__main__":
    if "--spatial-kind" not in sys.argv: sys.argv += ["--spatial-kind", "gt_masks"]
    raise SystemExit(main())
