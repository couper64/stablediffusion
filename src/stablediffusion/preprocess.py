"""CLI to build ``metadata.jsonl`` captions from a class-folder dataset.

Expects a standard image-classification layout::

    data/PetImages/
        Cat/0.jpg
        Dog/0.jpg

Example::

    stablediffusion-preprocess \\
        --data-dir data/PetImages \\
        --caption-template "a photo of a {class}"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .dataset import IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Append metadata.jsonl captions for a class-folder image dataset."
    )
    p.add_argument("--data-dir", required=True, help="Dataset root with one folder per class.")
    p.add_argument(
        "--output",
        default=None,
        help="Path to metadata.jsonl (default: <data-dir>/metadata.jsonl).",
    )
    p.add_argument(
        "--caption-template",
        default="a photo of a {class}",
        help='Caption template; ``{class}`` is the lowercased class folder name.',
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite metadata.jsonl from scratch instead of appending new images only.",
    )
    return p.parse_args()


def iter_classification_images(data_dir: Path) -> Iterable[Tuple[Path, Path]]:
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(data_dir)
        if len(rel.parts) < 2:
            continue
        yield path, rel


def caption_for_class(class_name: str, template: str) -> str:
    return template.replace("{class}", class_name.lower())


def build_entries(data_dir: Path, caption_template: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for _, rel in iter_classification_images(data_dir):
        class_name = rel.parts[0]
        entries.append(
            {
                "file_name": rel.as_posix(),
                "text": caption_for_class(class_name, caption_template),
            }
        )
    return entries


def load_existing_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    entries: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            file_name = entry.get("file_name")
            if file_name:
                entries[file_name] = entry
    return entries


def write_metadata(path: Path, entries: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_metadata(
    data_dir: Path,
    output_path: Path,
    caption_template: str,
    *,
    overwrite: bool = False,
) -> Tuple[int, int]:
    scanned = build_entries(data_dir, caption_template)
    if overwrite:
        write_metadata(output_path, scanned)
        return len(scanned), len(scanned)

    existing = load_existing_metadata(output_path)
    new_entries = [entry for entry in scanned if entry["file_name"] not in existing]
    if new_entries:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            for entry in new_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(scanned), len(new_entries)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    output_path = Path(args.output) if args.output else data_dir / "metadata.jsonl"
    total, added = append_metadata(
        data_dir=data_dir,
        output_path=output_path,
        caption_template=args.caption_template,
        overwrite=args.overwrite,
    )
    if total == 0:
        raise RuntimeError(
            f"No class-folder images found under {data_dir}. "
            "Expected layout like Cat/0.jpg with class names as subfolders."
        )

    action = "Wrote" if args.overwrite else "Appended"
    print(f"Scanned {total} image(s); {action.lower()} {added} entr{'y' if added == 1 else 'ies'} to {output_path}")


if __name__ == "__main__":
    main()
