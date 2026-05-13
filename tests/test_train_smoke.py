"""Integration smoke test: fine-tune LoRA on a small image subset."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from stablediff.train import main as train_main
from stablediff.util import load_lora_checkpoint

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for training smoke test"),
]


def test_train_on_image_sample(sample_image_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "smoke.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stablediff-train",
            "--data-dir",
            str(sample_image_dir),
            "--output",
            str(output_path),
            "--default-caption",
            "a photo of a cat",
            "--epochs",
            "1",
            "--rank",
            "4",
            "--val-split",
            "0.1",
            "--batch-size",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--num-workers",
            "0",
            "--resolution",
            "256",
            "--mixed-precision",
            "fp16",
        ],
    )

    train_main()

    assert output_path.is_file()
    checkpoint = load_lora_checkpoint(output_path)
    assert checkpoint["base_model"] == "runwayml/stable-diffusion-v1-5"
    assert checkpoint["lora_config"]["r"] == 4
    assert checkpoint["lora_state_dict"]
    assert checkpoint["meta"]["epoch"] == 1
    assert checkpoint["meta"]["resolution"] == 256
