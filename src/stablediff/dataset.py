"""Image-caption dataset for Stable Diffusion fine-tuning.

Layout supported under ``root``:

* image files (``.jpg/.jpeg/.png/.webp/.bmp``) anywhere under ``root``;
* an optional sidecar caption ``image.txt`` next to each image;
* an optional ``metadata.jsonl`` at ``root`` with lines of the form
  ``{"file_name": "relative/path.png", "text": "caption"}``.

When no caption is found for an image, ``default_caption`` is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        root: str,
        tokenizer,
        resolution: int = 512,
        default_caption: str = "",
        center_crop: bool = True,
        random_flip: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        self.tokenizer = tokenizer
        self.default_caption = default_caption

        self.captions_by_name: Dict[str, str] = {}
        metadata = self.root / "metadata.jsonl"
        if metadata.exists():
            with metadata.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    name = entry.get("file_name")
                    text = entry.get("text")
                    if name and text is not None:
                        self.captions_by_name[name] = text

        self.samples: List[Path] = sorted(
            p
            for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

        crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        steps = [
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            crop,
        ]
        if random_flip:
            steps.append(transforms.RandomHorizontalFlip())
        steps.extend([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.samples)

    def _caption_for(self, path: Path) -> str:
        rel = str(path.relative_to(self.root))
        if rel in self.captions_by_name:
            return self.captions_by_name[rel]
        if path.name in self.captions_by_name:
            return self.captions_by_name[path.name]
        sidecar = path.with_suffix(".txt")
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8").strip()
        return self.default_caption

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.samples[idx]
        image = Image.open(path).convert("RGB")
        pixel_values = self.transform(image)
        caption = self._caption_for(path)
        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }
