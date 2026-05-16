"""CLI to fine-tune Stable Diffusion with LoRA and save the best model as ``*.pt``.

Example::

    stablediffusion-train \\
        --data-dir data/my_set \\
        --output checkpoints/best.pt \\
        --base-model runwayml/stable-diffusion-v1-5 \\
        --epochs 30 --rank 8 --learning-rate 1e-4

If a ``val-split`` > 0 is provided, the held-out split's MSE is used to track
the best checkpoint. Otherwise the average training loss per epoch is used.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .dataset import ImageCaptionDataset
from .util import save_lora, suppress_known_upstream_warnings

DEFAULT_TARGET_MODULES: List[str] = ["to_q", "to_k", "to_v", "to_out.0"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Stable Diffusion with LoRA.")
    p.add_argument("--data-dir", required=True, help="Folder with training images (and optional captions).")
    p.add_argument("--output", required=True, help="Path to save the best LoRA checkpoint (*.pt).")
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5",
                   help="HuggingFace id or local path of the base SD model.")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank r.")
    p.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha (defaults to rank).")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction (0..1) of the dataset held out for validation.")
    p.add_argument("--default-caption", default="",
                   help="Caption to use when an image has no sidecar or metadata entry.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--save-every-epoch", action="store_true",
                   help="Also save a checkpoint after every epoch (suffixed with _epochN).")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    return p.parse_args()


def _pick_dtype(precision: str, device: str) -> torch.dtype:
    if device != "cuda" or precision == "no":
        return torch.float32
    return torch.float16 if precision == "fp16" else torch.bfloat16


def main() -> None:
    suppress_known_upstream_warnings()
    args = parse_args()
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = _pick_dtype(args.mixed_precision, device)
    use_amp = device == "cuda" and amp_dtype != torch.float32
    use_scaler = use_amp and amp_dtype == torch.float16
    frozen_dtype = amp_dtype if use_amp else torch.float32

    print(f"Device: {device} | mixed precision: {args.mixed_precision}")

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    text_encoder.to(device, dtype=frozen_dtype)
    vae.to(device, dtype=frozen_dtype)
    unet.to(device)

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.rank
    lora_config_kwargs = {
        "r"                 : args.rank,
        "lora_alpha"        : lora_alpha,
        "init_lora_weights" : "gaussian",
        "target_modules"    : DEFAULT_TARGET_MODULES,
    }
    unet.add_adapter(LoraConfig(**lora_config_kwargs))
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters found; check target_modules.")
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    dataset = ImageCaptionDataset(
        root=args.data_dir,
        tokenizer=tokenizer,
        resolution=args.resolution,
        default_caption=args.default_caption,
    )
    print(f"Dataset size: {len(dataset)} images.")

    if 0.0 < args.val_split < 1.0 and len(dataset) > 1:
        val_size = max(1, int(round(len(dataset) * args.val_split)))
        val_size = min(val_size, len(dataset) - 1)
        train_size = len(dataset) - val_size
        gen = torch.Generator().manual_seed(args.seed)
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)
    else:
        train_set, val_set = dataset, None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_set is not None
        else None
    )
    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty; reduce --batch-size or --val-split.")

    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def encode_text(input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return text_encoder(input_ids.to(device))[0]

    def encode_images(pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = vae.encode(pixel_values.to(device, dtype=frozen_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        return latents.to(torch.float32)

    def compute_loss(batch, deterministic: bool = False, step: int = 0) -> torch.Tensor:
        latents = encode_images(batch["pixel_values"])
        if deterministic:
            gen = torch.Generator(device=device).manual_seed(args.seed + 10_000 + step)
            noise = torch.randn(latents.shape, generator=gen, device=device)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                generator=gen,
                device=device,
            ).long()
        else:
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = encode_text(batch["input_ids"])

        prediction_type = noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return F.mse_loss(model_pred.float(), target.float())

    @torch.no_grad()
    def run_validation() -> float:
        if val_loader is None:
            return float("nan")
        unet.eval()
        losses: List[float] = []
        for i, batch in enumerate(val_loader):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = compute_loss(batch, deterministic=True, step=i)
            losses.append(loss.item())
        return sum(losses) / max(1, len(losses))

    best_metric = math.inf
    best_kind = "val_loss" if val_loader is not None else "train_loss"

    for epoch in range(args.epochs):
        unet.train()
        epoch_losses: List[float] = []
        accum_step = 0
        optimizer.zero_grad()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = compute_loss(batch)
            scaled_loss = loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accum_step += 1
            epoch_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if accum_step % args.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        if accum_step % args.gradient_accumulation_steps != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        train_mean = sum(epoch_losses) / max(1, len(epoch_losses))
        val_mean = run_validation() if val_loader is not None else float("nan")
        tracked = val_mean if val_loader is not None else train_mean
        print(
            f"Epoch {epoch + 1}: train_loss={train_mean:.4f} "
            f"val_loss={'n/a' if math.isnan(val_mean) else f'{val_mean:.4f}'} "
            f"[tracking {best_kind}={tracked:.4f}]"
        )

        if tracked < best_metric:
            best_metric = tracked
            save_lora(
                unet=unet,
                path=output_path,
                lora_config_kwargs=lora_config_kwargs,
                base_model=args.base_model,
                meta={
                    "epoch"       : epoch + 1,
                    "train_loss"  : train_mean,
                    "val_loss"    : None if math.isnan(val_mean) else val_mean,
                    "best_metric" : best_metric,
                    "metric_kind" : best_kind,
                    "resolution"  : args.resolution,
                },
            )
            print(f"  -> new best {best_kind}={best_metric:.4f}; saved to {output_path}")

        if args.save_every_epoch:
            ckpt_path = output_path.with_name(
                f"{output_path.stem}_epoch{epoch + 1}{output_path.suffix}"
            )
            save_lora(
                unet=unet,
                path=ckpt_path,
                lora_config_kwargs=lora_config_kwargs,
                base_model=args.base_model,
                meta={"epoch": epoch + 1, "train_loss": train_mean},
            )

    print(f"Training complete. Best {best_kind}: {best_metric:.4f}. Weights: {output_path}")


if __name__ == "__main__":
    main()
