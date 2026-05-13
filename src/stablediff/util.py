"""Shared utilities, including save/load helpers for LoRA adapter weights in ``*.pt`` format.

A LoRA checkpoint is a single ``torch.save`` dict with keys:

* ``lora_state_dict`` -- mapping returned by ``peft.utils.get_peft_model_state_dict``.
* ``lora_config`` -- kwargs needed to reconstruct ``peft.LoraConfig``.
* ``base_model`` -- HuggingFace id or local path of the Stable Diffusion model.
* ``meta`` -- optional free-form dict (training metrics, resolution, ...).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict


def suppress_known_upstream_warnings() -> None:
    """Silence deprecation noise from dependencies until upstream removes it."""
    warnings.filterwarnings(
        "ignore",
        message=r"The `local_dir_use_symlinks` argument is deprecated",
        category=UserWarning,
        module=r"huggingface_hub\.utils\._validators",
    )


def save_lora(
    unet,
    path: Path,
    lora_config_kwargs: Dict[str, Any],
    base_model: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = get_peft_model_state_dict(unet)
    payload: Dict[str, Any] = {
        "lora_state_dict": state_dict,
        "lora_config": lora_config_kwargs,
        "base_model": base_model,
    }
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)


def load_lora_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def attach_lora(unet, checkpoint: Dict[str, Any]) -> LoraConfig:
    config_kwargs = dict(checkpoint["lora_config"])
    lora_config = LoraConfig(**config_kwargs)
    unet.add_adapter(lora_config)
    set_peft_model_state_dict(unet, checkpoint["lora_state_dict"])
    return lora_config
