#!/usr/bin/env python3
"""Cache official AdapterShadow predictions for source/baseline image pairs.

The official backend deliberately imports AdapterShadow (and torch) lazily.  This
keeps manifest validation and the deterministic mock backend usable on CPU-only
machines that do not have the research repository's exact environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


OUTPUT_SIZE = 480
IMAGE_CACHE_KEYS = (
    "final_logit",
    "final_prob",
    "coarse_logit",
    "coarse_prob",
)
DELTA_CACHE_KEYS = ("positive_delta_final_prob", "positive_delta_coarse_prob")
IMAGE_SCHEMA = "adaptershadow_image_v1"
DELTA_SCHEMA = "adaptershadow_positive_delta_v1"


@dataclass(frozen=True)
class ManifestItem:
    manifest_index: int
    scene_id: str
    light_id: int | None
    source_path: Path
    target_path: Path
    target_cache: Path
    source_cache: Path
    delta_cache: Path


@dataclass(frozen=True)
class OfficialAssets:
    root: Path
    checkpoint: Path
    sam_checkpoint: Path
    efficientnet_checkpoint: Path


class CacheError(RuntimeError):
    """A user-facing cache runner error."""


def _resolve(path: str | Path, *, relative_to: Path | None = None) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() and relative_to is not None:
        value = relative_to / value
    return value.resolve()


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def official_asset_paths(args: argparse.Namespace) -> OfficialAssets:
    root = _resolve(args.adaptershadow_root)
    checkpoint = _resolve(args.checkpoint or root / "checkpoint/sbu.ckpt")
    sam_checkpoint = _resolve(
        args.sam_checkpoint or root / "checkpoint/sam/sam_vit_b_01ec64.pth"
    )
    efficientnet_checkpoint = _resolve(
        args.efficientnet_checkpoint
        or root
        / "efficientnet/hub/checkpoints/tf_efficientnet_b1_ap-44ef0a3d.pth"
    )
    return OfficialAssets(root, checkpoint, sam_checkpoint, efficientnet_checkpoint)


def validate_official_assets(assets: OfficialAssets) -> None:
    """Fail before importing torch or constructing the official model."""

    expected_efficientnet = (
        assets.root
        / "efficientnet/hub/checkpoints/tf_efficientnet_b1_ap-44ef0a3d.pth"
    ).resolve()
    required_repo_files = {
        "AdapterShadow model builder": assets.root / "models_exp/sam/build_sam.py",
        "local EfficientNet hub": (
            assets.root
            / "efficientnet/hub/rwightman_gen-efficientnet-pytorch_master/hubconf.py"
        ),
    }
    missing: list[str] = []
    if not _nonempty_file(assets.checkpoint):
        missing.append(
            f"AdapterShadow task checkpoint: {assets.checkpoint}\n"
            "    Expected the official SBU checkpoint (normally checkpoint/sbu.ckpt)."
        )
    if not _nonempty_file(assets.sam_checkpoint):
        missing.append(
            f"SAM ViT-B checkpoint: {assets.sam_checkpoint}\n"
            "    Expected sam_vit_b_01ec64.pth, matching the official vit_b builder."
        )
    if assets.efficientnet_checkpoint != expected_efficientnet:
        missing.append(
            "EfficientNet checkpoint path is incompatible with the unmodified official "
            f"loader: {assets.efficientnet_checkpoint}\n"
            f"    Place/use it at the hard-coded TORCH_HOME location: {expected_efficientnet}"
        )
    elif not _nonempty_file(assets.efficientnet_checkpoint):
        archive_hint = assets.root / "official_drive/AdapterShadow.zip"
        hint = (
            f" The downloaded archive exists at {archive_hint}; extract its "
            "AdapterShadow/code/efficientnet directory into the repository root."
            if _nonempty_file(archive_hint)
            else ""
        )
        missing.append(
            f"EfficientNet-B1 checkpoint: {assets.efficientnet_checkpoint}\n    {hint.strip()}"
        )
    for label, path in required_repo_files.items():
        if not _nonempty_file(path):
            missing.append(f"{label}: {path}")

    if missing:
        detail = "\n  - ".join(missing)
        raise CacheError(
            "Official AdapterShadow prerequisites are incomplete. No model was loaded.\n"
            f"  - {detail}\n"
            "The Google Drive archive distributed with this checkout may contain only "
            "the EfficientNet-B1 ImageNet weights; the AdapterShadow SBU and SAM "
            "checkpoints must be obtained separately."
        )


def _official_cache_id(assets: OfficialAssets) -> str:
    stamps = []
    for path in (
        assets.checkpoint,
        assets.sam_checkpoint,
        assets.efficientnet_checkpoint,
    ):
        stat = path.stat()
        stamps.append((str(path), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps(
        {
            "recipe": "sbu-b1-image_adapter-all-grid16-hw-v1",
            "assets": stamps,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "official-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _official_args(assets: OfficialAssets, batch_size: int) -> SimpleNamespace:
    """Exact effective arguments of the official README's SBU evaluation recipe."""

    # cfg.py defaults are included because the model modules access the Namespace
    # directly.  Effective non-default README flags are: npts=5, backbone=b1,
    # plug_image_adapter, all, freeze_backbone, use_neg_points, sample=grid,
    # grid_out_size=16.  The README spells use_neg_point in one place, while cfg.py
    # and the model consistently expose the plural use_neg_points attribute.
    return SimpleNamespace(
        net="sam",
        gpus="",
        sam_ckpt=str(assets.sam_checkpoint),
        val_freq=2,
        exp_name="sbu",
        pretrain=False,
        dataset_name="sbu",
        backbone="b1",
        logger="tensorboard",
        amp=False,
        use_neg_points=True,
        plug_dense_adapter=False,
        plug_image_adapter=True,
        plug_image_mask=False,
        plug_encoder_feature=False,
        plug_decoder_feature=False,
        plug_features_fusion=False,
        plug_idx=[2, 4, 6, 8, 10],
        freeze_backbone=True,
        vpt=False,
        small_size=False,
        skip_adapter=False,
        lora=False,
        multi_branch=False,
        mb_ratio=0.25,
        sample="grid",
        npts=5,
        grid_out_size=16,
        grid_thres=0.9,
        ratio_bt=1.0,
        a1=False,
        a2=False,
        all=True,
        tba=False,
        local_vit=False,
        down_ratio=0.25,
        bs=batch_size,
        thd=False,
        vis=None,
        image_size=1024,
        mask_size=256,
        mode="test",
        type="gen_pt",
        weights=str(assets.checkpoint),
        out_dir="",
        thres=128,
        save=False,
    )


class OfficialAdapterShadowBackend:
    """Thin, inference-only reproduction of ``pl_test_simple.py``."""

    name = "official"

    def __init__(
        self,
        assets: OfficialAssets,
        device: str,
        batch_size: int,
        output_size: int = OUTPUT_SIZE,
    ) -> None:
        validate_official_assets(assets)
        self.assets = assets
        self.output_size = output_size
        self.args = _official_args(assets, batch_size)
        self.cache_id = _official_cache_id(assets)

        try:
            import torch
            import torch.nn.functional as torch_f
        except Exception as exc:  # pragma: no cover - exercised only with official deps
            raise CacheError(
                "Could not import PyTorch for the official AdapterShadow backend. "
                "Activate/install the official repository environment, or use "
                "--backend mock for cache-pipeline testing."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise CacheError(
                f"Requested --device {device!r}, but torch.cuda.is_available() is false."
            )
        try:
            resolved_device = torch.device(device)
        except Exception as exc:
            raise CacheError(f"Invalid torch device {device!r}: {exc}") from exc

        repo_text = str(assets.root)
        inserted = repo_text not in sys.path
        if inserted:
            sys.path.insert(0, repo_text)
        try:
            with _working_directory(assets.root):
                try:
                    from models_exp.sam import sam_model_registry
                except Exception as exc:
                    raise CacheError(
                        "Failed to import the official AdapterShadow model modules from "
                        f"{assets.root}. Use the environment required by its README."
                    ) from exc

                try:
                    net = sam_model_registry["vit_b"](
                        self.args, checkpoint=str(assets.sam_checkpoint)
                    )
                except Exception as exc:
                    raise CacheError(
                        "The official AdapterShadow/SAM model could not be constructed. "
                        "The EfficientNet implementation uses a local torch.hub checkout "
                        "and its exact pretrained weight filename."
                    ) from exc
        finally:
            if inserted:
                try:
                    sys.path.remove(repo_text)
                except ValueError:
                    pass

        # Reproduce pl_test_simple.py: keep Adapter parameters trainable in the
        # image encoder, freeze the rest, and freeze the coarse EfficientNet when
        # freeze_backbone is enabled.  Freezing is not required for numerical
        # inference, but mirrors the official model state exactly.
        for parameter_name, parameter in net.image_encoder.named_parameters():
            if "Adapter" not in parameter_name:
                parameter.requires_grad = False
        if self.args.freeze_backbone:
            for parameter in net.mask_generator.efficient_encoder.parameters():
                parameter.requires_grad = False

        try:
            # Lightning task checkpoints contain more than tensors on some torch
            # versions, so weights_only=False preserves the official torch.load
            # behavior explicitly.
            try:
                checkpoint = torch.load(
                    assets.checkpoint, map_location="cpu", weights_only=False
                )
            except TypeError:  # torch < 2.0
                checkpoint = torch.load(assets.checkpoint, map_location="cpu")
            state = checkpoint["state_dict"]
            model_state = {
                key[6:]: value for key, value in state.items() if key.startswith("model.")
            }
            if not model_state:
                raise KeyError("no state_dict keys beginning with 'model.'")
            net.load_state_dict(model_state)
        except Exception as exc:
            raise CacheError(
                f"Could not load the official task checkpoint {assets.checkpoint}. "
                "Expected a Lightning checkpoint with state_dict keys prefixed by "
                f"'model.': {exc}"
            ) from exc

        self.torch = torch
        self.torch_f = torch_f
        self.device = resolved_device
        self.net = net.eval().to(self.device)

    def _load_batch(self, paths: Sequence[Path]):
        # Official dataloader.py divides RGB pixels by 255, then cv2.resize uses
        # default INTER_LINEAR to make the fixed 1024x1024 input.
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - official environment only
            raise CacheError(
                "The official preprocessing path requires OpenCV (cv2), as used by "
                "AdapterShadow's dataloader.py."
            ) from exc

        arrays: list[np.ndarray] = []
        for path in paths:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
            resized = cv2.resize(rgb, (1024, 1024))
            arrays.append(np.transpose(resized.astype(np.float32), (2, 0, 1)))
        return self.torch.from_numpy(np.stack(arrays)).to(self.device)

    def _resize(self, tensor):
        return self.torch_f.interpolate(
            tensor,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )

    def predict(self, paths: Sequence[Path]) -> list[dict[str, np.ndarray]]:
        if not paths:
            return []
        torch = self.torch
        images = self._load_batch(paths)
        with torch.inference_mode():
            # This is the official type=gen_pt + sample=grid path.  In
            # pl_test_simple.py grid_thres is applied to *raw coarse logits*.
            coarse_logit_native = self.net.mask_generator(images, images.shape[-2:])
            grid_pool = torch.nn.AdaptiveMaxPool2d(
                (self.args.grid_out_size, self.args.grid_out_size),
                return_indices=True,
            )
            values, indices = grid_pool(coarse_logit_native)
            values = values.flatten(1)
            indices = indices.flatten(2)
            mask_width = coarse_logit_native.shape[-1]
            hs = torch.div(indices, mask_width, rounding_mode="floor")
            ws = indices % mask_width
            # Intentionally retain official (h, w) concatenation/order.
            coords = torch.cat((hs, ws), dim=1).permute(0, 2, 1)
            labels = (values > self.args.grid_thres).float()
            sparse_embeddings, dense_embeddings = self.net.prompt_encoder(
                points=(coords, labels), boxes=None, masks=None
            )
            image_embeddings = self.net.image_encoder(images, None)
            final_logit_native, _ = self.net.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # Store both continuous fields at the requested cache resolution.
            # Probabilities are resized independently, matching the official
            # coarse path's sigmoid-before-resize ordering.
            tensors = {
                "final_logit": self._resize(final_logit_native),
                "final_prob": self._resize(torch.sigmoid(final_logit_native)),
                "coarse_logit": self._resize(coarse_logit_native),
                "coarse_prob": self._resize(torch.sigmoid(coarse_logit_native)),
            }
            cpu = {
                key: value[:, 0].detach().float().cpu().numpy()
                for key, value in tensors.items()
            }
        return [
            {key: cpu[key][index] for key in IMAGE_CACHE_KEYS}
            for index in range(len(paths))
        ]


class MockAdapterShadowBackend:
    """Dependency-free deterministic stand-in used to test the cache pipeline."""

    name = "mock"

    def __init__(self, output_size: int = OUTPUT_SIZE, seed: int = 0) -> None:
        self.output_size = output_size
        self.seed = int(seed)
        self.cache_id = f"mock-v1-seed-{self.seed}"
        axis = np.linspace(-1.0, 1.0, output_size, dtype=np.float32)
        self._xx, self._yy = np.meshgrid(axis, axis)

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)

    def predict(self, paths: Sequence[Path]) -> list[dict[str, np.ndarray]]:
        results: list[dict[str, np.ndarray]] = []
        phase = (self.seed % 997) / 997.0 * (2.0 * math.pi)
        fixed_pattern = 0.08 * np.sin(
            2.3 * self._xx - 1.7 * self._yy + phase
        ).astype(np.float32)
        for path in paths:
            with Image.open(path) as image:
                resampling = getattr(Image, "Resampling", Image)
                image = image.convert("RGB").resize(
                    (self.output_size, self.output_size), resampling.BILINEAR
                )
                rgb = np.asarray(image, dtype=np.float32) / 255.0
            luma = (
                0.2126 * rgb[..., 0]
                + 0.7152 * rgb[..., 1]
                + 0.0722 * rgb[..., 2]
            )
            coarse_logit = (
                4.0 * (0.52 - luma) + 0.12 * self._xx + fixed_pattern
            ).astype(np.float32)
            final_logit = (
                coarse_logit
                + 0.45 * (rgb[..., 0] - rgb[..., 2])
                + 0.05 * self._yy
            ).astype(np.float32)
            results.append(
                {
                    "final_logit": final_logit,
                    "final_prob": self._sigmoid(final_logit),
                    "coarse_logit": coarse_logit,
                    "coarse_prob": self._sigmoid(coarse_logit),
                }
            )
        return results


def _read_manifest(path: Path, limit: int) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CacheError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise CacheError(f"Expected a JSON object in {path}:{line_number}")
            if row.get("valid", True) is False:
                continue
            rows.append((line_number - 1, row))
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise CacheError(f"No valid manifest rows selected from {path}")
    return rows


def _safe_stem(value: str) -> str:
    value = Path(value).name
    stem = Path(value).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not safe:
        raise CacheError(f"Could not derive a safe cache filename from {value!r}")
    return safe


def _prediction_filename(row: Mapping[str, Any], prediction_name_key: str) -> str:
    if prediction_name_key:
        value = row.get(prediction_name_key)
        if not isinstance(value, str) or not value.strip():
            raise CacheError(
                f"Manifest row is missing non-empty --prediction-name-key "
                f"{prediction_name_key!r}"
            )
        name = Path(value).name
        # Match scripts/infer_manifest.py's output-name contract exactly.
        if not Path(name).suffix:
            name += ".png"
        return name
    scene_id = row.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        raise CacheError("Manifest row is missing string scene_id")
    light_id = row.get("light_id")
    if light_id is None:
        return f"{scene_id}.png"
    try:
        light_number = int(light_id)
    except (TypeError, ValueError) as exc:
        raise CacheError(f"Invalid light_id {light_id!r} for scene {scene_id}") from exc
    return f"{scene_id}_light_{light_number:03d}.png"


def _row_image_path(
    value: Any, *, key: str, relative_to: Path, manifest_index: int
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CacheError(
            f"Manifest row {manifest_index} is missing non-empty image key {key!r}"
        )
    return _resolve(value, relative_to=relative_to)


def build_manifest_items(args: argparse.Namespace) -> list[ManifestItem]:
    manifest = _resolve(args.manifest)
    if not manifest.is_file():
        raise CacheError(f"Manifest does not exist: {manifest}")
    base_path = _resolve(args.base_path or manifest.parent)
    baseline_dir = _resolve(args.baseline_dir) if args.baseline_dir else None
    output_dir = _resolve(args.output_dir)
    rows = _read_manifest(manifest, args.limit)

    items: list[ManifestItem] = []
    missing: list[str] = []
    cache_owners: dict[Path, int] = {}
    for manifest_index, row in rows:
        source_path = _row_image_path(
            row.get(args.source_key),
            key=args.source_key,
            relative_to=base_path,
            manifest_index=manifest_index,
        )
        prediction_name = _prediction_filename(row, args.prediction_name_key)
        if args.baseline_key:
            relative_to = baseline_dir or base_path
            target_path = _row_image_path(
                row.get(args.baseline_key),
                key=args.baseline_key,
                relative_to=relative_to,
                manifest_index=manifest_index,
            )
        else:
            if baseline_dir is None:
                raise CacheError(
                    "--baseline-dir is required when --baseline-key is not provided"
                )
            target_path = (baseline_dir / prediction_name).resolve()

        scene_id = str(row.get("scene_id", ""))
        light_raw = row.get("light_id")
        light_id = int(light_raw) if light_raw is not None else None
        target_stem = _safe_stem(prediction_name)
        target_cache = output_dir / "targets" / f"{target_stem}.npz"
        delta_cache = output_dir / "deltas" / f"{target_stem}.npz"
        source_digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:16]
        source_cache = (
            output_dir
            / "sources"
            / f"{source_digest}_{_safe_stem(source_path.name)}.npz"
        )
        for cache_path in (target_cache, delta_cache):
            owner = cache_owners.get(cache_path)
            if owner is not None:
                raise CacheError(
                    f"Cache filename collision between manifest rows {owner} and "
                    f"{manifest_index}: {cache_path.name}. Set --prediction-name-key "
                    "to a unique manifest field."
                )
            cache_owners[cache_path] = manifest_index
        if not source_path.is_file():
            missing.append(f"row {manifest_index} source: {source_path}")
        if not target_path.is_file():
            missing.append(f"row {manifest_index} baseline prediction: {target_path}")
        items.append(
            ManifestItem(
                manifest_index=manifest_index,
                scene_id=scene_id,
                light_id=light_id,
                source_path=source_path,
                target_path=target_path,
                target_cache=target_cache,
                source_cache=source_cache,
                delta_cache=delta_cache,
            )
        )

    if missing:
        preview = "\n  - ".join(missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise CacheError(
            "Source/baseline manifest preflight failed; no inference was started.\n"
            f"  - {preview}{suffix}"
        )
    return items


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix=f".{path.stem}.", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _cache_is_valid(
    path: Path,
    schema: str,
    required_keys: Sequence[str],
    *,
    cache_id: str,
    cache_dtype: np.dtype,
) -> bool:
    if not _nonempty_file(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not set(required_keys).issubset(archive.files):
                return False
            metadata_keys = {"schema", "output_size", "cache_id", "cache_dtype"}
            if not metadata_keys.issubset(archive.files):
                return False
            if str(archive["schema"].item()) != schema:
                return False
            if int(archive["output_size"].item()) != OUTPUT_SIZE:
                return False
            if str(archive["cache_id"].item()) != cache_id:
                return False
            if str(archive["cache_dtype"].item()) != cache_dtype.name:
                return False
    except (OSError, ValueError, KeyError, EOFError):
        return False
    return True


def _image_cache_valid(path: Path, cache_id: str, cache_dtype: np.dtype) -> bool:
    return _cache_is_valid(
        path,
        IMAGE_SCHEMA,
        IMAGE_CACHE_KEYS,
        cache_id=cache_id,
        cache_dtype=cache_dtype,
    )


def _delta_cache_valid(path: Path, cache_id: str, cache_dtype: np.dtype) -> bool:
    return _cache_is_valid(
        path,
        DELTA_SCHEMA,
        DELTA_CACHE_KEYS,
        cache_id=cache_id,
        cache_dtype=cache_dtype,
    )


def _cast_maps(
    prediction: Mapping[str, np.ndarray], cache_dtype: np.dtype, cache_id: str
) -> dict[str, Any]:
    arrays: dict[str, Any] = {
        "schema": np.asarray(IMAGE_SCHEMA),
        "output_size": np.asarray(OUTPUT_SIZE, dtype=np.int32),
        "cache_id": np.asarray(cache_id),
        "cache_dtype": np.asarray(cache_dtype.name),
    }
    for key in IMAGE_CACHE_KEYS:
        if key not in prediction:
            raise CacheError(f"Backend result is missing {key!r}")
        value = np.asarray(prediction[key])
        if value.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
            raise CacheError(
                f"Backend {key} has shape {value.shape}; expected "
                f"({OUTPUT_SIZE}, {OUTPUT_SIZE})"
            )
        if not np.isfinite(value).all():
            raise CacheError(f"Backend {key} contains NaN or infinity")
        arrays[key] = value.astype(cache_dtype, copy=False)
    return arrays


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _predict_to_cache(
    backend: Any,
    path_cache_pairs: Sequence[tuple[Path, Path]],
    *,
    batch_size: int,
    cache_dtype: np.dtype,
    cache_id: str,
) -> int:
    written = 0
    for chunk in _chunks(path_cache_pairs, batch_size):
        paths = [pair[0] for pair in chunk]
        predictions = backend.predict(paths)
        if len(predictions) != len(paths):
            raise CacheError(
                f"Backend returned {len(predictions)} results for {len(paths)} images"
            )
        for (_, cache_path), prediction in zip(chunk, predictions):
            _atomic_npz(cache_path, _cast_maps(prediction, cache_dtype, cache_id))
            written += 1
    return written


def _load_probability_maps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        final_prob = np.asarray(archive["final_prob"], dtype=np.float32)
        coarse_prob = np.asarray(archive["coarse_prob"], dtype=np.float32)
    return final_prob, coarse_prob


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cache AdapterShadow source/baseline logits, probabilities, and positive "
            "probability deltas at 480x480."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--base-path", default="", help="Root for relative source paths (manifest dir by default)."
    )
    parser.add_argument(
        "--baseline-dir",
        default="",
        help="Directory containing baseline predictions; required without --baseline-key.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-key", default="input_image")
    parser.add_argument(
        "--baseline-key",
        default="",
        help=(
            "Optional manifest field containing each baseline path. Relative values use "
            "--baseline-dir, or --base-path when no baseline directory is given."
        ),
    )
    parser.add_argument(
        "--prediction-name-key",
        default="",
        help=(
            "Optional manifest field used for baseline/cache filenames. Otherwise use "
            "scene_id[_light_NNN].png."
        ),
    )
    parser.add_argument("--backend", choices=("official", "mock"), default="official")
    parser.add_argument(
        "--adaptershadow-root", default="/workspace/external/AdapterShadow"
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--sam-checkpoint", default="")
    parser.add_argument("--efficientnet-checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, default=0, help="Maximum valid manifest rows; 0 means all."
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete atomic caches (default: true).",
    )
    parser.add_argument(
        "--cache-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--mock-seed", type=int, default=0)
    parser.add_argument("--index-name", default="index.jsonl")
    parser.add_argument(
        "--output-size",
        type=int,
        default=OUTPUT_SIZE,
        help="Fixed by this cache schema; only 480 is accepted.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, int]:
    if args.batch_size <= 0:
        raise CacheError("--batch-size must be positive")
    if args.limit < 0:
        raise CacheError("--limit must be non-negative")
    if args.output_size != OUTPUT_SIZE:
        raise CacheError(
            f"This cache contract requires --output-size {OUTPUT_SIZE}; got {args.output_size}."
        )

    items = build_manifest_items(args)
    output_dir = _resolve(args.output_dir)
    cache_dtype = np.dtype(args.cache_dtype)

    if args.backend == "official":
        assets = official_asset_paths(args)
        validate_official_assets(assets)
        cache_id = _official_cache_id(assets)
    else:
        assets = None
        cache_id = f"mock-v1-seed-{int(args.mock_seed)}"

    # Source predictions are content-addressed/shared across all target lights.
    unique_sources: dict[Path, tuple[Path, Path]] = {}
    for item in items:
        unique_sources.setdefault(
            item.source_cache, (item.source_path, item.source_cache)
        )
    source_pairs = list(unique_sources.values())
    if args.skip_existing:
        source_pairs = [
            pair
            for pair in source_pairs
            if not _image_cache_valid(pair[1], cache_id, cache_dtype)
        ]

    target_pairs = [(item.target_path, item.target_cache) for item in items]
    if args.skip_existing:
        target_pairs = [
            pair
            for pair in target_pairs
            if not _image_cache_valid(pair[1], cache_id, cache_dtype)
        ]

    backend: Any | None = None
    if source_pairs or target_pairs:
        if args.backend == "official":
            assert assets is not None
            backend = OfficialAdapterShadowBackend(
                assets, args.device, args.batch_size, OUTPUT_SIZE
            )
        else:
            backend = MockAdapterShadowBackend(OUTPUT_SIZE, args.mock_seed)
    source_written = _predict_to_cache(
        backend,
        source_pairs,
        batch_size=args.batch_size,
        cache_dtype=cache_dtype,
        cache_id=cache_id,
    )
    target_written = _predict_to_cache(
        backend,
        target_pairs,
        batch_size=args.batch_size,
        cache_dtype=cache_dtype,
        cache_id=cache_id,
    )

    delta_written = 0
    for item in items:
        if args.skip_existing and _delta_cache_valid(
            item.delta_cache, cache_id, cache_dtype
        ):
            continue
        source_final, source_coarse = _load_probability_maps(item.source_cache)
        target_final, target_coarse = _load_probability_maps(item.target_cache)
        delta_arrays = {
            "schema": np.asarray(DELTA_SCHEMA),
            "output_size": np.asarray(OUTPUT_SIZE, dtype=np.int32),
            "cache_id": np.asarray(cache_id),
            "cache_dtype": np.asarray(cache_dtype.name),
            "positive_delta_final_prob": np.maximum(
                target_final - source_final, 0.0
            ).astype(cache_dtype),
            "positive_delta_coarse_prob": np.maximum(
                target_coarse - source_coarse, 0.0
            ).astype(cache_dtype),
        }
        _atomic_npz(item.delta_cache, delta_arrays)
        delta_written += 1

    records: list[dict[str, Any]] = []
    for item in items:
        if not (
            _image_cache_valid(item.source_cache, cache_id, cache_dtype)
            and _image_cache_valid(item.target_cache, cache_id, cache_dtype)
            and _delta_cache_valid(item.delta_cache, cache_id, cache_dtype)
        ):
            raise CacheError(
                f"Postflight cache validation failed for manifest row {item.manifest_index}"
            )
        source_cache = _relative(item.source_cache, output_dir)
        target_cache = _relative(item.target_cache, output_dir)
        delta_cache = _relative(item.delta_cache, output_dir)
        records.append(
            {
                "manifest_index": item.manifest_index,
                "scene_id": item.scene_id,
                "light_id": item.light_id,
                "source_image": str(item.source_path),
                "baseline_prediction": str(item.target_path),
                "source_cache": source_cache,
                "target_cache": target_cache,
                "positive_delta_cache": delta_cache,
                "adapter_source_cache": source_cache,
                "adapter_target_cache": target_cache,
                "adapter_delta_cache": delta_cache,
                "backend": args.backend,
                "cache_id": cache_id,
                "output_size": OUTPUT_SIZE,
                "cache_dtype": args.cache_dtype,
            }
        )
    index_text = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    )
    _atomic_text(output_dir / args.index_name, index_text)
    summary = {
        "rows": len(items),
        "unique_sources": len(unique_sources),
        "source_caches_written": source_written,
        "target_caches_written": target_written,
        "delta_caches_written": delta_written,
    }
    print(json.dumps(summary, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (CacheError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
