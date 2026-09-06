from __future__ import annotations

import argparse
import json
import math
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.utils.data import save_video

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.infer_tokenlight import (  # noqa: E402
    encode_image_latents,
    extract_light_state,
    extract_lora_state,
    extract_type_state,
    infer_light_encoder_shape,
    infer_type_embedding_num_types,
    load_attrs,
    load_pipe,
    load_state,
)
from model.lightoken_encoder import LightokenEncoder  # noqa: E402
from model.tokenlight_wan import (  # noqa: E402
    TOKENLIGHT_TYPE_LIGHT,
    TOKENLIGHT_TYPE_MASK,
    TOKENLIGHT_TYPE_SOURCE,
    TOKENLIGHT_TYPE_TARGET,
    TokenLightTypeEmbedding,
    _add_type_embedding,
    _clean_prefix_t_mod,
    _freqs_for_grid,
    _patch_to_tokens,
    gradient_checkpoint_forward_compatible,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TokenLight inference with step decodes and light-token self-attention dumps.",
    )
    parser.add_argument("--weights_dir", default="weights/Wan2.2-TI2V-5B")
    parser.add_argument("--source", required=True)
    parser.add_argument("--attrs", required=True, help="Inline JSON or path to JSON attrs.")
    parser.add_argument("--mask", default="")
    parser.add_argument("--checkpoint", default="", help="Checkpoint containing LoRA and light_encoder weights.")
    parser.add_argument("--lora_checkpoint", default="")
    parser.add_argument("--light_checkpoint", default="")
    parser.add_argument("--prompt", default="photorealistic object relighting, preserve geometry and materials")
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token_dim", type=int, default=0)
    parser.add_argument("--fourier_features", type=int, default=512)
    parser.add_argument("--fourier_sigma", type=float, default=5.0)
    parser.add_argument("--tokenlight_max_lights", "--max_lights", type=int, default=1)
    parser.add_argument("--tokenlight_mask_tokens", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--debug_dir", default="", help="Directory for step decodes and attention maps.")
    parser.add_argument("--save_step_decodes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_step_latents", action="store_true")
    parser.add_argument("--step_decode_steps", default="all", help="Comma/range list, e.g. all or 0,5,10-20.")
    parser.add_argument("--save_attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_steps", default="all", help="Comma/range list, e.g. all or 0,5,10-20.")
    parser.add_argument("--attention_layers", default="all", help="Comma/range list, e.g. all or 0,8,16,24.")
    parser.add_argument(
        "--attention_directions",
        default="target_to_light,source_to_light,light_to_target,light_to_source",
        help="Comma list from target_to_light,source_to_light,light_to_target,light_to_source.",
    )
    parser.add_argument("--attention_phases", default="pos", help="Comma list from pos,neg. Default saves conditioned pass.")
    parser.add_argument("--attention_query_chunk_size", type=int, default=256)
    parser.add_argument("--save_attention_png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention_png_width", type=int, default=0, help="0 means use --width.")
    parser.add_argument("--attention_png_height", type=int, default=0, help="0 means use --height.")
    return parser.parse_args()


def parse_index_selector(value: str) -> set[int] | None:
    value = str(value or "all").strip().lower()
    if value in {"all", "*"}:
        return None
    if value in {"none", ""}:
        return set()
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                start, end = end, start
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def selected(selector: set[int] | None, value: int) -> bool:
    return selector is None or int(value) in selector


def debug_dir_for(args: argparse.Namespace) -> Path:
    if args.debug_dir:
        return Path(args.debug_dir)
    output = Path(args.output)
    return output.parent / f"{output.stem}_debug"


@dataclass(frozen=True)
class TokenLayout:
    source: tuple[int, int] | None
    mask: tuple[int, int] | None
    light: tuple[int, int] | None
    target: tuple[int, int]
    source_grid: tuple[int, int, int] | None
    mask_grid: tuple[int, int, int] | None
    target_grid: tuple[int, int, int]
    light_names: tuple[str, ...]


class AttentionRecorder:
    def __init__(self, args: argparse.Namespace, debug_dir: Path) -> None:
        self.debug_dir = debug_dir
        self.raw_dir = debug_dir / "attention_pt"
        self.png_dir = debug_dir / "attention_png"
        self.step_selector = parse_index_selector(args.attention_steps)
        self.layer_selector = parse_index_selector(args.attention_layers)
        self.enabled = bool(args.save_attention)
        self.phases = {item.strip() for item in str(args.attention_phases).split(",") if item.strip()}
        self.directions = {item.strip() for item in str(args.attention_directions).split(",") if item.strip()}
        self.query_chunk_size = max(1, int(args.attention_query_chunk_size))
        self.save_png = bool(args.save_attention_png)
        self.png_size = (
            int(args.attention_png_width) if int(args.attention_png_width) > 0 else int(args.width),
            int(args.attention_png_height) if int(args.attention_png_height) > 0 else int(args.height),
        )
        self.current_step: int | None = None
        self.current_timestep: int | None = None
        self.current_phase: str | None = None
        self.layout: TokenLayout | None = None

    @contextmanager
    def forward_context(self, step: int, timestep: int, phase: str):
        previous = (self.current_step, self.current_timestep, self.current_phase, self.layout)
        self.current_step = int(step)
        self.current_timestep = int(timestep)
        self.current_phase = str(phase)
        self.layout = None
        try:
            yield
        finally:
            self.current_step, self.current_timestep, self.current_phase, self.layout = previous

    def set_layout(self, layout: TokenLayout) -> None:
        self.layout = layout

    def should_capture(self, layer_idx: int) -> bool:
        if not self.enabled:
            return False
        if self.current_step is None or self.current_phase is None or self.layout is None:
            return False
        if self.current_phase not in self.phases:
            return False
        return selected(self.step_selector, self.current_step) and selected(self.layer_selector, layer_idx)

    @torch.no_grad()
    def capture(self, *, layer_idx: int, q: torch.Tensor, k: torch.Tensor, num_heads: int) -> None:
        if not self.should_capture(layer_idx):
            return
        layout = self.layout
        if layout is None or layout.light is None:
            return
        light_start, light_end = layout.light
        light_indices = list(range(light_start, light_end))
        if not light_indices:
            return

        q_heads = self._split_heads(q.detach(), num_heads)
        k_heads = self._split_heads(k.detach(), num_heads)
        payload: dict[str, Any] = {
            "step": self.current_step,
            "timestep": self.current_timestep,
            "phase": self.current_phase,
            "layer": int(layer_idx),
            "segments": {
                "source": layout.source,
                "mask": layout.mask,
                "light": layout.light,
                "target": layout.target,
            },
            "grids": {
                "source": layout.source_grid,
                "mask": layout.mask_grid,
                "target": layout.target_grid,
            },
            "light_names": layout.light_names,
            "directions": {},
        }

        if "target_to_light" in self.directions:
            payload["directions"]["target_to_light"] = self._query_to_light(q_heads, k_heads, layout.target, light_indices)
        if "source_to_light" in self.directions and layout.source is not None:
            payload["directions"]["source_to_light"] = self._query_to_light(q_heads, k_heads, layout.source, light_indices)
        if "light_to_target" in self.directions:
            payload["directions"]["light_to_target"] = self._light_to_key(q_heads, k_heads, light_indices, layout.target)
        if "light_to_source" in self.directions and layout.source is not None:
            payload["directions"]["light_to_source"] = self._light_to_key(q_heads, k_heads, light_indices, layout.source)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        name = f"step_{self.current_step:03d}_layer_{layer_idx:02d}_{self.current_phase}.pt"
        torch.save(payload, self.raw_dir / name)
        if self.save_png:
            self._save_payload_pngs(payload)

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        if dim % num_heads != 0:
            raise ValueError(f"Cannot split dim={dim} into {num_heads} heads")
        head_dim = dim // num_heads
        return x.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).float()

    def _query_to_light(
        self,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        query: tuple[int, int],
        light_indices: list[int],
    ) -> torch.Tensor:
        start, end = query
        total = max(0, end - start)
        output = torch.empty(
            q_heads.shape[0],
            len(light_indices),
            total,
            dtype=torch.float32,
            device="cpu",
        )
        key_indices = torch.tensor(light_indices, dtype=torch.long, device=q_heads.device)
        scale = 1.0 / math.sqrt(float(q_heads.shape[-1]))
        write_offset = 0
        k_t = k_heads.transpose(-2, -1)
        for chunk_start in range(start, end, self.query_chunk_size):
            chunk_end = min(end, chunk_start + self.query_chunk_size)
            logits = torch.matmul(q_heads[:, :, chunk_start:chunk_end], k_t) * scale
            probs = logits.softmax(dim=-1)
            maps = probs.index_select(dim=-1, index=key_indices).mean(dim=1)
            maps = maps.permute(0, 2, 1).contiguous().cpu()
            output[:, :, write_offset : write_offset + maps.shape[-1]] = maps
            write_offset += maps.shape[-1]
        return output

    def _light_to_key(
        self,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        light_indices: list[int],
        key: tuple[int, int],
    ) -> torch.Tensor:
        key_start, key_end = key
        output = torch.empty(
            q_heads.shape[0],
            len(light_indices),
            max(0, key_end - key_start),
            dtype=torch.float32,
            device="cpu",
        )
        scale = 1.0 / math.sqrt(float(q_heads.shape[-1]))
        k_t = k_heads.transpose(-2, -1)
        for out_idx, light_idx in enumerate(light_indices):
            logits = torch.matmul(q_heads[:, :, light_idx : light_idx + 1], k_t) * scale
            probs = logits.softmax(dim=-1)
            output[:, out_idx] = probs[:, :, 0, key_start:key_end].mean(dim=1).contiguous().cpu()
        return output

    def _save_payload_pngs(self, payload: dict[str, Any]) -> None:
        grids = payload["grids"]
        direction_grids = {
            "target_to_light": grids["target"],
            "source_to_light": grids["source"],
            "light_to_target": grids["target"],
            "light_to_source": grids["source"],
        }
        for direction, maps in payload["directions"].items():
            grid = direction_grids.get(direction)
            if grid is None:
                continue
            for token_idx, token_name in enumerate(payload["light_names"]):
                safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(token_name))
                base = (
                    self.png_dir
                    / f"step_{payload['step']:03d}"
                    / f"layer_{payload['layer']:02d}"
                    / str(direction)
                    / f"light_{token_idx:02d}_{safe_name}"
                )
                save_spatial_map_pngs(maps[0, token_idx], grid, base, self.png_size)


def save_spatial_map_pngs(values: torch.Tensor, grid: tuple[int, int, int], base: Path, image_size: tuple[int, int]) -> None:
    frames, height, width = [int(item) for item in grid]
    expected = frames * height * width
    if int(values.numel()) != expected:
        return
    data = values.detach().float().reshape(frames, height, width)
    base.parent.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(frames):
        frame = data[frame_idx]
        finite = torch.isfinite(frame)
        if finite.any():
            lo = frame[finite].min()
            hi = frame[finite].max()
            norm = (frame - lo) / (hi - lo + 1e-8)
        else:
            norm = torch.zeros_like(frame)
        arr = (norm.clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().numpy()
        image = Image.fromarray(arr, mode="L").resize(image_size, resample=Image.Resampling.BILINEAR)
        suffix = f"_f{frame_idx:03d}.png" if frames > 1 else ".png"
        image.save(base.with_name(base.name + suffix))


def install_self_attention_recorder(dit: torch.nn.Module, recorder: AttentionRecorder) -> None:
    for layer_idx, block in enumerate(getattr(dit, "blocks", [])):
        attn = getattr(getattr(block, "self_attn", None), "attn", None)
        if attn is None:
            continue
        if hasattr(attn, "_tokenlight_debug_original_forward"):
            attn._tokenlight_debug_recorder = recorder
            attn._tokenlight_debug_layer_idx = int(layer_idx)
            continue
        attn._tokenlight_debug_original_forward = attn.forward
        attn._tokenlight_debug_recorder = recorder
        attn._tokenlight_debug_layer_idx = int(layer_idx)

        def debug_forward(self, q, k, v):
            out = self._tokenlight_debug_original_forward(q, k, v)
            recorder_obj = getattr(self, "_tokenlight_debug_recorder", None)
            if recorder_obj is not None:
                recorder_obj.capture(
                    layer_idx=int(getattr(self, "_tokenlight_debug_layer_idx", -1)),
                    q=q,
                    k=k,
                    num_heads=int(self.num_heads),
                )
            return out

        attn.forward = types.MethodType(debug_forward, attn)


def model_fn_wan_video_tokenlight_debug(
    *,
    attention_recorder: AttentionRecorder | None,
    dit: torch.nn.Module,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    clip_feature: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    motion_controller: torch.nn.Module | None = None,
    motion_bucket_id: torch.Tensor | None = None,
    tokenlight_light_encoder: LightokenEncoder | None = None,
    tokenlight_type_embedding: TokenLightTypeEmbedding | None = None,
    tokenlight_attrs=None,
    tokenlight_drop_light=False,
    tokenlight_source_latents: torch.Tensor | None = None,
    tokenlight_mask_latents: torch.Tensor | None = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
) -> torch.Tensor:
    del kwargs
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
    from einops import rearrange

    if getattr(dit, "seperated_timestep", False) and fuse_vae_embedding_in_latents:
        timestep = torch.concat(
            [
                torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
                torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep,
            ]
        ).flatten()
        t_head = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t_head).unflatten(2, (6, dit.dim))
    else:
        t_head = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t_head).unflatten(1, (6, dit.dim))

    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)
    batch = context.shape[0]
    x = latents if latents.shape[0] == batch else torch.cat([latents] * batch, dim=0)
    if y is not None and getattr(dit, "require_vae_embedding", True):
        x = torch.cat([x, _repeat_to_batch(y, batch)], dim=1)
    if clip_feature is not None and getattr(dit, "require_clip_embedding", True):
        x_clip = _repeat_to_batch(clip_feature, batch)
        context = torch.cat([dit.img_emb(x_clip), context], dim=1)

    patches = dit.patchify(x, control_camera_latents_input)
    target_grid = patches.shape[2:]
    target_tokens = rearrange(patches, "b c f h w -> b (f h w) c").contiguous()
    target_tokens = _add_type_embedding(target_tokens, tokenlight_type_embedding, TOKENLIGHT_TYPE_TARGET)
    target_freqs = _freqs_for_grid(dit, target_grid, target_tokens.device)

    prefix_tokens: list[torch.Tensor] = []
    prefix_freqs: list[torch.Tensor] = []
    cursor = 0
    source_slice = None
    mask_slice = None
    light_slice = None
    source_grid = None
    mask_grid = None

    if tokenlight_source_latents is not None:
        source_tokens, source_grid = _patch_to_tokens(dit, tokenlight_source_latents, batch)
        source_tokens = _add_type_embedding(source_tokens, tokenlight_type_embedding, TOKENLIGHT_TYPE_SOURCE)
        source_slice = (cursor, cursor + source_tokens.shape[1])
        cursor += source_tokens.shape[1]
        prefix_tokens.append(source_tokens)
        prefix_freqs.append(_freqs_for_grid(dit, source_grid, target_tokens.device))
    if tokenlight_mask_latents is not None:
        mask_tokens, mask_grid = _patch_to_tokens(dit, tokenlight_mask_latents, batch)
        mask_tokens = _add_type_embedding(mask_tokens, tokenlight_type_embedding, TOKENLIGHT_TYPE_MASK)
        mask_slice = (cursor, cursor + mask_tokens.shape[1])
        cursor += mask_tokens.shape[1]
        prefix_tokens.append(mask_tokens)
        prefix_freqs.append(_freqs_for_grid(dit, mask_grid, target_tokens.device))
    light_names: tuple[str, ...] = ()
    if tokenlight_light_encoder is not None:
        light_tokens = tokenlight_light_encoder(
            tokenlight_attrs,
            batch_size=batch,
            device=target_tokens.device,
            dtype=target_tokens.dtype,
            drop_light=tokenlight_drop_light,
        )
        light_names = tuple(getattr(tokenlight_light_encoder, "token_names", ())) or tuple(
            f"light_{index:02d}" for index in range(light_tokens.shape[1])
        )
        light_tokens = _add_type_embedding(light_tokens, tokenlight_type_embedding, TOKENLIGHT_TYPE_LIGHT)
        light_slice = (cursor, cursor + light_tokens.shape[1])
        cursor += light_tokens.shape[1]
        prefix_tokens.append(light_tokens)
        prefix_freqs.append(torch.ones(light_tokens.shape[1], 1, target_freqs.shape[-1], device=target_tokens.device, dtype=target_freqs.dtype))

    if prefix_tokens:
        prefix_len = sum(tokens.shape[1] for tokens in prefix_tokens)
        x = torch.cat([*prefix_tokens, target_tokens], dim=1)
        freqs = torch.cat([*prefix_freqs, target_freqs], dim=0)
        if t_mod.ndim == 4:
            clean_t_mod = _clean_prefix_t_mod(dit, prefix_len, t_mod.shape[0], t_mod.dtype, t_mod.device)
            t_mod = torch.cat([clean_t_mod, t_mod], dim=1)
    else:
        prefix_len = 0
        x = target_tokens
        freqs = target_freqs

    if attention_recorder is not None:
        attention_recorder.set_layout(
            TokenLayout(
                source=source_slice,
                mask=mask_slice,
                light=light_slice,
                target=(prefix_len, prefix_len + target_tokens.shape[1]),
                source_grid=tuple(source_grid) if source_grid is not None else None,
                mask_grid=tuple(mask_grid) if mask_grid is not None else None,
                target_grid=tuple(target_grid),
                light_names=light_names,
            )
        )

    for block in dit.blocks:
        x = gradient_checkpoint_forward_compatible(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            x,
            context,
            t_mod,
            freqs,
        )

    x = x[:, prefix_len:] if prefix_len else x
    x = dit.head(x, t_head)
    return dit.unpatchify(x, target_grid)


def _repeat_to_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError(f"Cannot expand batch {tensor.shape[0]} to {batch}")
    return tensor.expand(batch, *tensor.shape[1:]).contiguous()


def base_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input_image": None,
        "end_image": None,
        "input_video": None,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": 1 / 54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1,
        "seed": args.seed,
        "rand_device": args.device,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "cfg_scale": 1,
        "cfg_merge": False,
        "sigma_shift": 5.0,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": True,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": 16000,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
        "vap_video": None,
        "vap_prompt": " ",
        "negative_vap_prompt": " ",
        "wantodance_music_path": None,
        "wantodance_reference_image": None,
        "wantodance_fps": 30,
        "wantodance_keyframes": None,
        "wantodance_keyframes_mask": None,
        "framewise_decoding": False,
    }


def save_video_or_image(frames, output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        frames[0].save(output)
    else:
        save_video(frames, str(output), fps=fps, quality=5)


@torch.no_grad()
def decode_saved_steps(pipe, step_latents: list[dict[str, Any]], args: argparse.Namespace, debug_dir: Path) -> None:
    if not step_latents:
        return
    out_dir = debug_dir / "step_decodes"
    latent_dir = debug_dir / "step_latents"
    step_selector = parse_index_selector(args.step_decode_steps)
    pipe.load_models_to_device(["vae"])
    for item in tqdm(step_latents, desc="Decoding saved steps"):
        step = int(item["step"])
        if not selected(step_selector, step):
            continue
        latents = item["latents"].to(device=pipe.device, dtype=pipe.torch_dtype)
        video = pipe.vae.decode(latents, device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
        frames = pipe.vae_output_to_video(video)
        suffix = ".png" if len(frames) == 1 else ".mp4"
        save_video_or_image(frames, out_dir / f"step_{step:03d}{suffix}", fps=args.fps)
        if args.save_step_latents:
            latent_dir.mkdir(parents=True, exist_ok=True)
            torch.save(item, latent_dir / f"step_{step:03d}.pt")


@torch.no_grad()
def generate_debug(
    pipe,
    light_encoder: LightokenEncoder,
    type_embedding: TokenLightTypeEmbedding | None,
    attrs: dict[str, float],
    source: Image.Image,
    mask: Image.Image | None,
    args: argparse.Namespace,
    debug_dir: Path,
):
    recorder = AttentionRecorder(args, debug_dir)
    install_self_attention_recorder(pipe.dit, recorder)

    pipe.model_fn = lambda **kwargs: model_fn_wan_video_tokenlight_debug(
        attention_recorder=recorder,
        tokenlight_light_encoder=light_encoder,
        tokenlight_type_embedding=type_embedding,
        **kwargs,
    )
    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=5.0)
    inputs_posi = {"prompt": args.prompt}
    inputs_nega = {"prompt": args.prompt}
    inputs_shared = base_inputs(args)
    inputs_shared["rand_device"] = pipe.device
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    inputs_shared["tokenlight_attrs"] = [attrs]
    source_latents = encode_image_latents(pipe, source, args)
    inputs_shared["tokenlight_source_latents"] = source_latents
    if mask is not None:
        inputs_shared["tokenlight_mask_latents"] = encode_image_latents(pipe, mask, args)
    elif getattr(args, "tokenlight_mask_tokens", True):
        inputs_shared["tokenlight_mask_latents"] = torch.zeros_like(source_latents)
    inputs_posi["tokenlight_drop_light"] = False
    inputs_nega["tokenlight_drop_light"] = True

    step_latents: list[dict[str, Any]] = []
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for index, timestep_raw in enumerate(tqdm(pipe.scheduler.timesteps, desc="Denoising")):
        timestep = timestep_raw.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        timestep_int = int(timestep_raw.detach().cpu().item())
        with recorder.forward_context(index, timestep_int, "pos"):
            noise_pos = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
        if args.cfg_scale != 1.0:
            with recorder.forward_context(index, timestep_int, "neg"):
                noise_neg = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
            noise = noise_neg + args.cfg_scale * (noise_pos - noise_neg)
        else:
            noise = noise_pos
        inputs_shared["latents"] = pipe.scheduler.step(noise, pipe.scheduler.timesteps[index], inputs_shared["latents"])
        if args.save_step_decodes or args.save_step_latents:
            step_latents.append(
                {
                    "step": int(index),
                    "timestep": timestep_int,
                    "latents": inputs_shared["latents"].detach().cpu(),
                }
            )

    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(inputs_shared["latents"], device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
    final_frames = pipe.vae_output_to_video(video)
    if args.save_step_decodes or args.save_step_latents:
        decode_saved_steps(pipe, step_latents, args, debug_dir)
    return final_frames


def write_debug_manifest(args: argparse.Namespace, debug_dir: Path, light_encoder: LightokenEncoder) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output": str(args.output),
        "debug_dir": str(debug_dir),
        "num_inference_steps": int(args.num_inference_steps),
        "cfg_scale": float(args.cfg_scale),
        "step_decode_steps": args.step_decode_steps,
        "attention_steps": args.attention_steps,
        "attention_layers": args.attention_layers,
        "attention_directions": args.attention_directions,
        "attention_phases": args.attention_phases,
        "light_token_names": list(getattr(light_encoder, "token_names", ())),
    }
    (debug_dir / "debug_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    debug_dir = debug_dir_for(args)
    pipe = load_pipe(args)
    combined = load_state(args.checkpoint)
    lora_state = load_state(args.lora_checkpoint) if args.lora_checkpoint else combined
    lora = extract_lora_state(lora_state)
    if lora:
        pipe.load_lora(pipe.dit, state_dict=lora, alpha=1.0)

    token_dim = args.token_dim if args.token_dim > 0 else int(pipe.dit.dim)
    light_checkpoint_state = load_state(args.light_checkpoint) if args.light_checkpoint else combined
    light_state = extract_light_state(light_checkpoint_state)
    max_lights, fourier_features = infer_light_encoder_shape(
        light_state,
        requested_max_lights=getattr(args, "tokenlight_max_lights", 1),
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
    type_state = extract_type_state(light_checkpoint_state)
    if type_state:
        num_types = infer_type_embedding_num_types(type_state, requested_num_types=4)
        type_embedding = TokenLightTypeEmbedding(token_dim, num_types=num_types).to(
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
        type_embedding.load_state_dict(type_state, strict=False)
        type_embedding.eval()

    source = Image.open(args.source).convert("RGB")
    mask = Image.open(args.mask).convert("RGB") if args.mask else None
    write_debug_manifest(args, debug_dir, light_encoder)
    video = generate_debug(pipe, light_encoder, type_embedding, load_attrs(args.attrs), source, mask, args, debug_dir)
    save_video_or_image(video, Path(args.output), fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
