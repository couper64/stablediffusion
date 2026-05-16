"""CLI to generate images with Stable Diffusion and (optionally) a trained LoRA.

Examples::

    # plain SD generation
    stablediffusion-generate --prompt "a photo of a corgi" --output outputs/

    # using a LoRA checkpoint produced by ``stablediffusion-train``
    stablediffusion-generate \\
        --lora checkpoints/best.pt \\
        --prompt "a portrait of <subject>" \\
        --num-images 4 --steps 30 --guidance-scale 7.5 --seed 1234
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline

from .util import (
    attach_lora,
    load_lora_checkpoint,
    sd_pipeline_load_kwargs,
    suppress_known_upstream_warnings,
)

DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate images with Stable Diffusion + optional LoRA.")
    p.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Text prompt. Pass multiple times for multiple prompts.",
    )
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--lora", default=None, help="Path to a LoRA checkpoint (*.pt) from stablediffusion-train.")
    p.add_argument("--lora-scale", type=float, default=1.0, help="Adapter scale applied at inference.")
    p.add_argument("--base-model", default=None,
                   help="Override base model id. Defaults to the checkpoint's base_model or SD 1.5.")
    p.add_argument("--output", default="outputs", help="Output directory.")
    p.add_argument("--num-images", type=int, default=1, help="Images per prompt.")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--enable-attention-slicing", action="store_true",
                   help="Reduce VRAM at the cost of speed.")
    p.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Do not load the NSFW safety checker (not recommended for public-facing use).",
    )
    return p.parse_args()


def _resolve_dtype(name: str, device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    dtypes = {
        "fp16" : torch.float16,
        "bf16" : torch.bfloat16,
        "fp32" : torch.float32,
    }
    return dtypes[name]


def main() -> None:
    suppress_known_upstream_warnings()
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(args.dtype, device)

    checkpoint = None
    base_model: Optional[str] = args.base_model
    if args.lora:
        checkpoint = load_lora_checkpoint(Path(args.lora))
        if base_model is None:
            base_model = checkpoint.get("base_model")
    if base_model is None:
        base_model = DEFAULT_BASE_MODEL

    print(f"Loading base model: {base_model} (device={device}, dtype={dtype})")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        **sd_pipeline_load_kwargs(disable_safety_checker=args.disable_safety_checker),
    )
    pipe.set_progress_bar_config(disable=False)
    pipe.to(device)

    if checkpoint is not None:
        print(f"Attaching LoRA from {args.lora}")
        attach_lora(pipe.unet, checkpoint)
        pipe.unet.to(device, dtype=dtype)
        try:
            pipe.unet.set_adapters(["default"], weights=[args.lora_scale])
        except (AttributeError, TypeError, ValueError):
            if args.lora_scale != 1.0:
                print(f"  (warning: could not apply lora-scale={args.lora_scale}; using 1.0)")

    if args.enable_attention_slicing:
        pipe.enable_attention_slicing()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    saved: List[Path] = []
    for p_idx, prompt in enumerate(args.prompt):
        print(f"[{p_idx + 1}/{len(args.prompt)}] {prompt!r}")
        result = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt or None,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images,
            height=args.height,
            width=args.width,
            generator=generator,
        )
        for i, image in enumerate(result.images):
            path = out_dir / f"prompt{p_idx:02d}_image{i:02d}.png"
            image.save(path)
            saved.append(path)
            print(f"  saved {path}")

    print(f"Generated {len(saved)} image(s) into {out_dir}")


if __name__ == "__main__":
    main()
