"""CLI to benchmark single-image inference time and energy use.

Example::

    stablediffusion-benchmark \\
        --prompt "a photo of a cat" \\
        --lora checkpoints/cats.pt \\
        --output metrics/benchmark.json
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from codecarbon import EmissionsTracker
from diffusers import StableDiffusionPipeline

from .util import attach_lora, load_lora_checkpoint, suppress_known_upstream_warnings, write_json_results

DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark inference latency and energy for one image.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--lora", default=None, help="Path to a LoRA checkpoint (*.pt) from stablediffusion-train.")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument("--base-model", default=None)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--warmup", action="store_true", help="Run one untimed warmup generation first.")
    p.add_argument("--output", required=True, help="Path to write JSON results.")
    p.add_argument("--enable-attention-slicing", action="store_true")
    return p.parse_args()


def _resolve_dtype(name: str, device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _load_pipeline(
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
    *,
    base_model: str,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> StableDiffusionPipeline:
    print(f"Loading base model: {base_model} (device={device}, dtype={dtype})")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.set_progress_bar_config(disable=True)
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
    return pipe


def _generate_one(
    pipe: StableDiffusionPipeline,
    args: argparse.Namespace,
    generator: Optional[torch.Generator],
) -> None:
    pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=1,
        height=args.height,
        width=args.width,
        generator=generator,
    )


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

    pipe = _load_pipeline(args, device, dtype, base_model=base_model, checkpoint=checkpoint)

    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.warmup:
        print("Running warmup generation...")
        _generate_one(pipe, args, generator)

    tracker = EmissionsTracker(project_name="stablediffusion-benchmark", save_to_file=False)
    tracker.start()
    start = time.perf_counter()
    _generate_one(pipe, args, generator)
    seconds_per_image = time.perf_counter() - start
    emissions_kg_co2 = tracker.stop()

    emissions: Dict[str, Any] = {}
    if tracker.final_emissions_data is not None:
        emissions = asdict(tracker.final_emissions_data)

    results = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "lora": args.lora,
        "base_model": base_model,
        "device": device,
        "dtype": args.dtype,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "seconds_per_image": seconds_per_image,
        "emissions_kg_co2": emissions_kg_co2,
        "energy_kwh": emissions.get("energy_consumed"),
        "codecarbon": emissions,
    }
    output_path = Path(args.output)
    write_json_results(output_path, results)
    print(f"Generated 1 image in {seconds_per_image:.3f}s")
    if emissions_kg_co2 is not None:
        print(f"Emissions: {emissions_kg_co2 * 1000:.3f} g CO2eq")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
