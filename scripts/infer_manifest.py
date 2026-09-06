#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_PROMPT = "photorealistic object relighting, preserve geometry and materials"

np = None
torch = None
Image = None
ImageDraw = None
ImageFont = None
tqdm = None

extract_light_state = None
extract_lora_state = None
extract_type_state = None
generate = None
load_pipe = None
load_state = None
infer_light_encoder_shape = None
infer_type_embedding_num_types = None
LightokenEncoder = None
TokenLightTypeEmbedding = None
parse_attrs_json = None


def ensure_runtime_imports(*, include_model: bool) -> None:
    global np, torch, Image, ImageDraw, ImageFont, tqdm
    global extract_light_state, extract_lora_state, extract_type_state, generate, load_pipe, load_state
    global infer_light_encoder_shape, infer_type_embedding_num_types
    global LightokenEncoder, TokenLightTypeEmbedding, parse_attrs_json

    if torch is None or Image is None or ImageFont is None:
        import numpy as _np
        import torch as _torch
        from PIL import Image as _Image
        from PIL import ImageDraw as _ImageDraw
        from PIL import ImageFont as _ImageFont
        from tqdm import tqdm as _tqdm

        np = _np
        torch = _torch
        Image = _Image
        ImageDraw = _ImageDraw
        ImageFont = _ImageFont
        tqdm = _tqdm

    if include_model and LightokenEncoder is None:
        from model.infer import (  # noqa: WPS433
            extract_light_state as _extract_light_state,
            extract_lora_state as _extract_lora_state,
            extract_type_state as _extract_type_state,
            generate as _generate,
            infer_light_encoder_shape as _infer_light_encoder_shape,
            infer_type_embedding_num_types as _infer_type_embedding_num_types,
            load_pipe as _load_pipe,
            load_state as _load_state,
        )
        from model.light_encoder import LightokenEncoder as _LightokenEncoder  # noqa: WPS433
        from model.light_encoder import parse_attrs_json as _parse_attrs_json  # noqa: WPS433
        from model.wan import TokenLightTypeEmbedding as _TokenLightTypeEmbedding  # noqa: WPS433

        extract_light_state = _extract_light_state
        extract_lora_state = _extract_lora_state
        extract_type_state = _extract_type_state
        generate = _generate
        infer_light_encoder_shape = _infer_light_encoder_shape
        infer_type_embedding_num_types = _infer_type_embedding_num_types
        load_pipe = _load_pipe
        load_state = _load_state
        LightokenEncoder = _LightokenEncoder
        TokenLightTypeEmbedding = _TokenLightTypeEmbedding
        parse_attrs_json = _parse_attrs_json


def add_bool_arg(parser: argparse.ArgumentParser, name: str, *, default: bool, help_text: str = "") -> None:
    option = name if name.startswith("--") else f"--{name}"
    dest = option[2:].replace("-", "_")
    parser.add_argument(option, dest=dest, action="store_true", help=help_text)
    parser.add_argument(f"--no-{option[2:]}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TokenLight inference from one manifest, optionally with metrics.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-path", default="", help="Base path for relative image paths inside the manifest.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="", help="Checkpoint containing LoRA, light encoder, and type embedding.")
    parser.add_argument("--weights_dir", default="weights/Wan2.2-TI2V-5B")

    parser.add_argument("--source-key", default="input_image")
    parser.add_argument("--target-key", default="target_image")
    parser.add_argument("--target-fallback-key", default="video")
    parser.add_argument("--mask-key", default="inf_mask")
    parser.add_argument("--mask-fallback-key", default="mask")
    parser.add_argument("--extra-mask-keys", default="")
    parser.add_argument("--eval-mask-key", default="mask")
    parser.add_argument("--attrs-key", default="attrs_json")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument(
        "--seed-key",
        default="",
        help="Optional manifest key containing a per-row inference seed.",
    )
    parser.add_argument(
        "--prediction-name-key",
        default="",
        help="Optional manifest key containing a unique output filename.",
    )

    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-devices",
        "--gpu_devices",
        default="",
        help="Comma-separated GPU ids for multi-process inference, e.g. 0,1,2,3. Use cpu for CPU.",
    )
    parser.add_argument("--token_dim", type=int, default=0)
    parser.add_argument("--fourier_features", type=int, default=512)
    parser.add_argument("--fourier_sigma", type=float, default=5.0)
    parser.add_argument("--tokenlight_max_lights", "--max-lights", type=int, default=1)
    add_bool_arg(parser, "--tokenlight_mask_tokens", default=True)
    add_bool_arg(parser, "--use-mask-input", default=False)
    add_bool_arg(
        parser,
        "--require-mask-input",
        default=False,
        help_text=(
            "Validate every resolved mask before model loading and refuse missing, "
            "wrong-sized, or non-binary mask inputs. Empty all-zero masks are valid."
        ),
    )

    add_bool_arg(parser, "--skip-existing", default=True)
    add_bool_arg(parser, "--with-gt", default=True)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--eval", action="store_true", help="Compute PSNR/SSIM/LPIPS after inference.")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing predictions.")
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--metric-device", default="auto")
    parser.add_argument("--lpips-net", default="alex")
    add_bool_arg(parser, "--allow-missing", default=True)
    return parser.parse_args()


def resolve_repo(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def data_base(args: argparse.Namespace) -> Path:
    return resolve_repo(args.base_path) if args.base_path else ROOT


def resolve_data(path: str | Path, args: argparse.Namespace) -> Path:
    value = Path(path)
    return value if value.is_absolute() else data_base(args) / value


def row_value(row: dict[str, Any], key: str, fallback_key: str = "") -> Any:
    value = row.get(key)
    if value not in (None, ""):
        return value
    return row.get(fallback_key) if fallback_key else None


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("valid") is False:
                continue
            row["_manifest_index"] = index
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def parse_gpu_devices(value: str | list[Any] | None) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        devices = [str(item).strip() for item in value if str(item).strip()]
    else:
        devices = [item.strip() for item in str(value).split(",") if item.strip()]
    normalized: list[str] = []
    for device in devices:
        lowered = device.lower()
        if lowered == "cpu":
            normalized.append("cpu")
            continue
        if lowered.startswith("cuda:"):
            device = device.split(":", 1)[1]
        if not device.isdigit():
            raise ValueError(f"Unsupported GPU device value: {device!r}")
        normalized.append(device)
    return normalized


def split_rows_by_device(rows: list[dict[str, Any]], devices: list[str]) -> list[tuple[str, list[dict[str, Any]]]]:
    buckets = {device: [] for device in devices}
    for index, row in enumerate(rows):
        buckets[devices[index % len(devices)]].append(row)
    return [(device, bucket) for device, bucket in buckets.items() if bucket]


def apply_worker_device(device_id: str, args: argparse.Namespace) -> argparse.Namespace:
    worker_args = argparse.Namespace(**vars(args))
    if device_id == "cpu":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        worker_args.device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        worker_args.device = "cuda"
    return worker_args


def prediction_path(output_dir: Path, row: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.prediction_name_key:
        value = row.get(args.prediction_name_key)
        if value in (None, ""):
            raise KeyError(
                f"Missing prediction name key {args.prediction_name_key!r} "
                f"in manifest row {row.get('_manifest_index')}"
            )
        name = Path(str(value)).name
        if not Path(name).suffix:
            name += ".png"
        return output_dir / name
    scene_id = str(row.get("scene_id") or f"item_{int(row.get('_manifest_index', 0)):06d}")
    light_id = row.get("light_id")
    if light_id is None:
        return output_dir / f"{scene_id}.png"
    return output_dir / f"{scene_id}_light_{int(light_id):03d}.png"


def with_gt_path(pred: Path) -> Path:
    return pred.with_name(f"{pred.stem}_withgt{pred.suffix}")


def target_path(row: dict[str, Any], args: argparse.Namespace) -> Path | None:
    value = row_value(row, args.target_key, args.target_fallback_key)
    return resolve_data(value, args) if value else None


def source_path(row: dict[str, Any], args: argparse.Namespace) -> Path:
    value = row.get(args.source_key)
    if not value:
        raise KeyError(f"Missing source key {args.source_key!r} in manifest row {row.get('_manifest_index')}")
    return resolve_data(value, args)


def mask_path(row: dict[str, Any], args: argparse.Namespace) -> Path | None:
    value = row_value(row, args.mask_key, args.mask_fallback_key)
    return resolve_data(value, args) if value else None


def _required_mask_value(row: dict[str, Any], args: argparse.Namespace) -> tuple[Any, str, bool]:
    """Resolve one required mask while preserving whether fallback was used."""
    mask_key = str(getattr(args, "mask_key", "") or "")
    if not mask_key:
        raise KeyError("--mask-key must be non-empty when --require-mask-input is enabled")
    value = row.get(mask_key)
    if value not in (None, ""):
        return value, mask_key, False

    fallback_key = str(getattr(args, "mask_fallback_key", "") or "")
    if not fallback_key:
        raise KeyError(f"Missing required mask key {mask_key!r}; mask fallback is disabled")
    fallback = row.get(fallback_key)
    if fallback in (None, ""):
        raise KeyError(
            f"Missing required mask key {mask_key!r} and fallback key {fallback_key!r}"
        )
    return fallback, fallback_key, True


def _validate_required_mask_asset(path: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    """Validate the hard white-foreground mask representation used during training."""
    from PIL import Image as PILImage

    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with PILImage.open(path) as opened:
            if opened.size != expected_size:
                raise ValueError(
                    f"expected size {expected_size[0]}x{expected_size[1]}, "
                    f"got {opened.size[0]}x{opened.size[1]}"
                )
            histogram = opened.convert("L").histogram()
    except (OSError, SyntaxError) as error:
        raise ValueError(f"cannot decode mask image: {error}") from error

    unexpected = [value for value, count in enumerate(histogram) if count and value not in (0, 255)]
    if unexpected:
        preview = unexpected[:8]
        suffix = "..." if len(unexpected) > len(preview) else ""
        raise ValueError(
            "mask must be hard binary with 0=background and 255=foreground (white); "
            f"found values {preview}{suffix}"
        )
    foreground_pixels = int(histogram[255])
    pixel_count = int(sum(histogram))
    return {
        "empty": foreground_pixels == 0,
        "foreground_pixels": foreground_pixels,
        "pixel_count": pixel_count,
    }


def preflight_required_masks(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Validate all required conditioning masks before any model is loaded."""
    if not bool(getattr(args, "use_mask_input", False)):
        raise ValueError("--require-mask-input requires --use-mask-input")
    if not bool(getattr(args, "tokenlight_mask_tokens", False)):
        raise ValueError("--require-mask-input requires --tokenlight_mask_tokens")

    expected_size = (int(args.width), int(args.height))
    if expected_size[0] <= 0 or expected_size[1] <= 0:
        raise ValueError(f"Invalid required mask size: {expected_size}")

    asset_results: dict[Path, dict[str, Any] | Exception] = {}
    failures: list[str] = []
    failure_count = 0
    primary_rows = 0
    fallback_rows = 0
    empty_rows = 0

    for row_index, row in enumerate(rows):
        manifest_index = row.get("_manifest_index", row_index)
        identity = f"row {manifest_index} ({row.get('scene_id', '<unknown>')}/{row.get('light_id', '<none>')})"
        try:
            value, _, used_fallback = _required_mask_value(row, args)
            path = resolve_data(value, args)
            if path not in asset_results:
                try:
                    asset_results[path] = _validate_required_mask_asset(path, expected_size)
                except Exception as error:  # retain one decoded result for duplicate paths
                    asset_results[path] = error
            result = asset_results[path]
            if isinstance(result, Exception):
                raise result
            fallback_rows += int(used_fallback)
            primary_rows += int(not used_fallback)
            empty_rows += int(bool(result["empty"]))
        except Exception as error:
            failure_count += 1
            if len(failures) < 20:
                failures.append(f"{identity}: {error}")

    if failure_count:
        details = "\n".join(f"  - {item}" for item in failures)
        omitted = failure_count - len(failures)
        if omitted:
            details += f"\n  - ... {omitted} additional error(s)"
        raise ValueError(f"Required mask preflight failed with {failure_count} error(s):\n{details}")

    summary = {
        "schema": "tokenlight_required_mask_preflight_v1",
        "row_count": len(rows),
        "unique_mask_count": len(asset_results),
        "primary_rows": primary_rows,
        "fallback_rows": fallback_rows,
        "empty_rows": empty_rows,
        "expected_size": [expected_size[0], expected_size[1]],
        "polarity": "0=background,255=foreground_white",
        "empty_masks_allowed": True,
    }
    print(
        "[infer_manifest] required-mask preflight: "
        f"rows={summary['row_count']} unique={summary['unique_mask_count']} "
        f"fallback={summary['fallback_rows']} empty={summary['empty_rows']} "
        f"size={expected_size[0]}x{expected_size[1]} polarity=white-foreground",
        flush=True,
    )
    return summary


def extra_mask_paths(row: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for key in (item.strip() for item in str(args.extra_mask_keys or "").split(",")):
        if not key:
            continue
        value = row.get(key)
        if value in (None, ""):
            raise KeyError(f"Missing extra mask key {key!r} in manifest row {row.get('_manifest_index')}")
        paths.append(resolve_data(value, args))
    return paths


def eval_mask_path(row: dict[str, Any], args: argparse.Namespace) -> Path | None:
    value = row.get(args.eval_mask_key)
    return resolve_data(value, args) if value not in (None, "") else None


def label_font(panel_height: int):
    font_size = max(14, min(18, panel_height // 30))
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_labeled_panels(panels: list[tuple[str, Any]], out_file: Path) -> None:
    if not panels:
        raise ValueError("No panels to render")
    width, height = panels[0][1].size
    font = label_font(height)
    label_height = max(34, int(getattr(font, "size", 14) * 2.2))
    canvas = Image.new("RGB", (width * len(panels), height + label_height), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = index * width
        canvas.paste(image, (x, 0))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(
            (
                x + width // 2 - text_width // 2 - bbox[0],
                height + label_height // 2 - text_height // 2 - bbox[1],
            ),
            label,
            fill=(245, 245, 245),
            font=font,
        )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_file)


def save_with_gt(pred_path: Path, source: Path | Image.Image, target: Path) -> None:
    pred = Image.open(pred_path).convert("RGB")
    if isinstance(source, Path):
        source_image = Image.open(source).convert("RGB")
    else:
        source_image = source.convert("RGB")
    panels = [
        ("source", source_image.resize(pred.size, Image.Resampling.BICUBIC)),
        ("output", pred),
        ("gt", Image.open(target).convert("RGB").resize(pred.size, Image.Resampling.BICUBIC)),
    ]
    render_labeled_panels(panels, with_gt_path(pred_path))


def setup_pipeline(args: argparse.Namespace):
    ensure_runtime_imports(include_model=True)
    pipe = load_pipe(args)
    combined = load_state(args.checkpoint)
    lora = extract_lora_state(combined)
    if lora:
        pipe.load_lora(pipe.dit, state_dict=lora, alpha=1.0)

    token_dim = args.token_dim if args.token_dim > 0 else int(pipe.dit.dim)
    light_state = extract_light_state(combined)
    max_lights, fourier_features = infer_light_encoder_shape(
        light_state,
        requested_max_lights=args.tokenlight_max_lights,
        requested_fourier_features=args.fourier_features,
    )
    light_encoder = LightokenEncoder(
        token_dim,
        fourier_features=fourier_features,
        fourier_sigma=args.fourier_sigma,
        max_lights=max_lights,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    if light_state:
        light_encoder.load_state_dict(light_state, strict=False)
    light_encoder.eval()

    type_embedding = None
    type_state = extract_type_state(combined)
    if type_state:
        num_types = infer_type_embedding_num_types(type_state, requested_num_types=4)
        type_embedding = TokenLightTypeEmbedding(token_dim, num_types=num_types).to(
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
        type_embedding.load_state_dict(type_state, strict=False)
        type_embedding.eval()
    return pipe, light_encoder, type_embedding


def attrs_from_row(row: dict[str, Any], key: str) -> dict[str, Any]:
    return parse_attrs_json(row.get(key))


def run_inference(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> int:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --eval-only is set")
    pipe, light_encoder, type_embedding = setup_pipeline(args)
    completed = 0
    desc = Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).stem
    for row in tqdm(rows, desc=desc):
        pred = prediction_path(output_dir, row, args)
        target = target_path(row, args)
        if args.skip_existing and pred.exists():
            if args.with_gt and target and target.exists() and not with_gt_path(pred).exists():
                save_with_gt(pred, source_path(row, args), target)
            completed += 1
            continue

        source = Image.open(source_path(row, args)).convert("RGB")
        mask = None
        current_mask = mask_path(row, args)
        if args.use_mask_input and current_mask and current_mask.exists():
            mask = Image.open(current_mask).convert("RGB")
        elif bool(getattr(args, "require_mask_input", False)):
            raise FileNotFoundError(
                current_mask
                or f"Required mask disappeared or no longer resolves for manifest row {row.get('_manifest_index')}"
            )
        current_extra_masks = extra_mask_paths(row, args)
        missing_extra_masks = [path for path in current_extra_masks if not path.exists()]
        if missing_extra_masks:
            raise FileNotFoundError(missing_extra_masks[0])
        extra_masks = [Image.open(path).convert("RGB") for path in current_extra_masks]

        infer_args = argparse.Namespace(**vars(args))
        infer_args.prompt = row.get(args.prompt_key) or args.prompt
        if args.seed_key:
            if row.get(args.seed_key) in (None, ""):
                raise KeyError(
                    f"Missing seed key {args.seed_key!r} in manifest row {row.get('_manifest_index')}"
                )
            infer_args.seed = int(row[args.seed_key])
        video = generate(
            pipe,
            light_encoder,
            type_embedding,
            attrs_from_row(row, args.attrs_key),
            source,
            mask,
            infer_args,
            extra_masks=extra_masks,
        )
        pred.parent.mkdir(parents=True, exist_ok=True)
        video[0].save(pred)
        if args.with_gt and target and target.exists():
            save_with_gt(pred, source, target)
        completed += 1
    return completed


def run_inference_worker(device_id: str, rows: list[dict[str, Any]], output_dir: str, args_dict: dict[str, Any]) -> int:
    args = apply_worker_device(device_id, argparse.Namespace(**args_dict))
    print(f"[infer_manifest:{device_id}] rows={len(rows)} device={args.device}", flush=True)
    completed = run_inference(rows, Path(output_dir), args)
    print(f"[infer_manifest:{device_id}] completed={completed}", flush=True)
    return completed


def run_inference_distributed(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> int:
    devices = parse_gpu_devices(args.gpu_devices)
    if not devices:
        return run_inference(rows, output_dir, args)
    assigned = split_rows_by_device(rows, devices)
    if len(assigned) == 1:
        device_id, shard = assigned[0]
        worker_args = apply_worker_device(device_id, args)
        print(f"[infer_manifest] gpu_devices={','.join(devices)} rows={len(rows)}", flush=True)
        return run_inference(shard, output_dir, worker_args)

    print(
        f"[infer_manifest] gpu_devices={','.join(devices)} processes={len(assigned)} rows={len(rows)}",
        flush=True,
    )
    args_dict = vars(args)
    completed = 0
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(assigned), mp_context=context) as executor:
        futures = [
            executor.submit(run_inference_worker, device_id, shard, output_dir.as_posix(), args_dict)
            for device_id, shard in assigned
        ]
        for future in concurrent.futures.as_completed(futures):
            completed += int(future.result())
    return completed


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def load_mask(path: Path, size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    array = (np.asarray(image, dtype=np.float32) / 255.0) > 0.5
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred.float() - target.float()).square())
    return float((-10.0 * torch.log10(mse + eps)).detach().cpu())


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> float:
    weights = mask.to(device=pred.device, dtype=pred.dtype).expand_as(pred)
    denom = torch.clamp(weights.sum(), min=1.0)
    mse = ((pred.float() - target.float()).square() * weights).sum() / denom
    return float((-10.0 * torch.log10(mse + eps)).detach().cpu())


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    from torch.nn import functional as F

    pred = pred.float()
    target = target.float()
    channels = pred.shape[1]
    padding = window_size // 2
    weight = torch.ones(channels, 1, window_size, window_size, device=pred.device) / (window_size * window_size)
    mu_x = F.conv2d(pred, weight, padding=padding, groups=channels)
    mu_y = F.conv2d(target, weight, padding=padding, groups=channels)
    sigma_x = F.conv2d(pred * pred, weight, padding=padding, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(target * target, weight, padding=padding, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(pred * target, weight, padding=padding, groups=channels) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return float(score.mean().detach().cpu())


def mask_bbox(mask: torch.Tensor) -> tuple[int, int, int, int]:
    ys, xs = torch.where(mask[0, 0] > 0.5)
    if ys.numel() == 0:
        return 0, 0, mask.shape[-1], mask.shape[-2]
    y0 = int(ys.min().item())
    y1 = int(ys.max().item()) + 1
    x0 = int(xs.min().item())
    x1 = int(xs.max().item()) + 1
    return x0, y0, x1, y1


def object_crop(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.nn import functional as F

    x0, y0, x1, y1 = mask_bbox(mask)
    pred_crop = pred[:, :, y0:y1, x0:x1]
    target_crop = target[:, :, y0:y1, x0:x1]
    mask_crop = mask[:, :, y0:y1, x0:x1].to(device=pred.device, dtype=pred.dtype)
    neutral = torch.full_like(pred_crop, 0.5)
    obj_pred = torch.where(mask_crop > 0.5, pred_crop, neutral)
    obj_target = torch.where(mask_crop > 0.5, target_crop, neutral)
    height, width = obj_pred.shape[-2:]
    min_side = 64
    if height < min_side or width < min_side:
        scale = max(min_side / max(1, height), min_side / max(1, width))
        size = (max(min_side, int(round(height * scale))), max(min_side, int(round(width * scale))))
        obj_pred = F.interpolate(obj_pred, size=size, mode="bilinear", align_corners=False)
        obj_target = F.interpolate(obj_target, size=size, mode="bilinear", align_corners=False)
    return obj_pred, obj_target


def background_images(
    pred: torch.Tensor, target: torch.Tensor, object_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Neutralize object pixels so SSIM/LPIPS measure only the background."""
    mask = object_mask.to(device=pred.device, dtype=pred.dtype)
    neutral = torch.full_like(pred, 0.5)
    return torch.where(mask > 0.5, neutral, pred), torch.where(mask > 0.5, neutral, target)


class LpipsMetric:
    def __init__(self, device: torch.device, net: str) -> None:
        import lpips

        self.model = lpips.LPIPS(net=net).to(device).eval()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self.model(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean().detach().cpu())


def average(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def average_metrics(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        metric: average([record[key][metric] for record in records])
        for metric in ("psnr", "ssim", "lpips")
    }


def metric_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_eval(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    ensure_runtime_imports(include_model=False)
    device = metric_device(args.metric_device)
    lpips_metric = LpipsMetric(device, args.lpips_net)
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    with torch.no_grad():
        for row in tqdm(rows, desc="metrics"):
            pred_file = prediction_path(output_dir, row, args)
            current_target = target_path(row, args)
            if not pred_file.exists():
                missing.append({"index": row.get("_manifest_index"), "pred": pred_file.as_posix()})
                if not args.allow_missing:
                    raise FileNotFoundError(pred_file)
                continue
            if current_target is None or not current_target.exists():
                missing.append({"index": row.get("_manifest_index"), "target": "" if current_target is None else current_target.as_posix()})
                if not args.allow_missing:
                    raise FileNotFoundError(current_target or "<missing target>")
                continue

            size = Image.open(pred_file).size
            pred = load_rgb(pred_file).to(device)
            target = load_rgb(current_target, size=size).to(device)
            current_mask = eval_mask_path(row, args)
            mask = load_mask(current_mask, size=size).to(device) if current_mask and current_mask.exists() else None
            has_object_mask = bool(mask is not None and (mask > 0.5).any().item())
            obj_pred, obj_target = object_crop(pred, target, mask) if has_object_mask else (None, None)
            bg_pred, bg_target = background_images(pred, target, mask) if has_object_mask else (None, None)
            record = {
                "index": row.get("_manifest_index"),
                "scene_id": row.get("scene_id"),
                "light_id": row.get("light_id"),
                "pred": pred_file.as_posix(),
                "target": current_target.as_posix(),
                "mask": "" if current_mask is None else current_mask.as_posix(),
                "full_image": {
                    "psnr": psnr(pred, target),
                    "ssim": ssim(pred, target),
                    "lpips": lpips_metric(pred, target),
                },
                "object_only": (
                    {
                        "psnr": masked_psnr(pred, target, mask),
                        "ssim": ssim(obj_pred, obj_target),
                        "lpips": lpips_metric(obj_pred, obj_target),
                    }
                    if has_object_mask
                    else {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}
                ),
                "background_only": (
                    {
                        "psnr": masked_psnr(pred, target, 1.0 - mask),
                        "ssim": ssim(bg_pred, bg_target),
                        "lpips": lpips_metric(bg_pred, bg_target),
                    }
                    if has_object_mask
                    else {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}
                ),
            }
            records.append(record)

    per_scene: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        per_scene.setdefault(str(record["scene_id"]), []).append(record)
    scene_averages = {
        scene_id: {
            "count": len(items),
            "full_image": average_metrics(items, "full_image"),
            "object_only": average_metrics(items, "object_only"),
            "background_only": average_metrics(items, "background_only"),
        }
        for scene_id, items in sorted(per_scene.items())
    }
    return {
        "summary": {
            "expected_count": len(rows),
            "evaluated_count": len(records),
            "missing_count": len(missing),
            "full_image": average_metrics(records, "full_image"),
            "object_only": average_metrics(records, "object_only"),
            "background_only": average_metrics(records, "background_only"),
        },
        "scene_averages": scene_averages,
        "missing": missing,
        "records": records,
    }


def write_snapshot(rows: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> None:
    snapshot = {
        "manifest": resolve_repo(args.manifest).as_posix(),
        "base_path": data_base(args).as_posix(),
        "output_dir": output_dir.as_posix(),
        "checkpoint": "" if not args.checkpoint else resolve_repo(args.checkpoint).as_posix(),
        "source_key": args.source_key,
        "target_key": args.target_key,
        "target_fallback_key": args.target_fallback_key,
        "mask_key": args.mask_key,
        "mask_fallback_key": args.mask_fallback_key,
        "extra_mask_keys": args.extra_mask_keys,
        "eval_mask_key": args.eval_mask_key,
        "tokenlight_mask_tokens": getattr(args, "tokenlight_mask_tokens", None),
        "require_mask_input": bool(getattr(args, "require_mask_input", False)),
        "required_mask_preflight": getattr(args, "required_mask_preflight", None),
        "attrs_key": args.attrs_key,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "tokenlight_max_lights": args.tokenlight_max_lights,
        "use_mask_input": args.use_mask_input,
        "gpu_devices": parse_gpu_devices(args.gpu_devices),
        "eval": args.eval,
        "eval_only": args.eval_only,
        "row_count": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "infer_manifest_config_resolved.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.manifest = resolve_repo(args.manifest).as_posix()
    args.weights_dir = resolve_repo(args.weights_dir).as_posix()
    if args.checkpoint:
        args.checkpoint = resolve_repo(args.checkpoint).as_posix()

    rows = load_rows(Path(args.manifest), int(args.limit))
    output_dir = resolve_repo(args.output_dir)
    if bool(getattr(args, "require_mask_input", False)):
        args.required_mask_preflight = preflight_required_masks(rows, args)
    write_snapshot(rows, output_dir, args)

    completed = 0
    if not args.eval_only:
        completed = run_inference_distributed(rows, output_dir, args)
        print(f"[infer_manifest] completed={completed} output_dir={output_dir}", flush=True)

    if args.eval or args.eval_only:
        metrics = run_eval(rows, output_dir, args)
        metrics_output = resolve_repo(args.metrics_output) if args.metrics_output else output_dir / "metrics.json"
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest": args.manifest,
            "base_path": data_base(args).as_posix(),
            "pred_dir": output_dir.as_posix(),
            "metrics_output": metrics_output.as_posix(),
            "device": str(metric_device(args.metric_device)),
            "lpips_net": args.lpips_net,
            **metrics,
        }
        metrics_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[infer_manifest] wrote metrics: {metrics_output}", flush=True)
        print(json.dumps(payload["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
