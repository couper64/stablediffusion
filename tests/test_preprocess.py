from __future__ import annotations

import json
from pathlib import Path

from stablediffusion.preprocess import append_metadata, build_entries, load_existing_metadata


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_build_entries_from_class_folders(tmp_path: Path) -> None:
    _touch(tmp_path / "Cat" / "0.jpg")
    _touch(tmp_path / "Dog" / "1.png")

    entries = build_entries(tmp_path, "a photo of a {class}")

    assert entries == [
        {"file_name": "Cat/0.jpg", "text": "a photo of a cat"},
        {"file_name": "Dog/1.png", "text": "a photo of a dog"},
    ]


def test_append_skips_existing_entries(tmp_path: Path) -> None:
    _touch(tmp_path / "Cat" / "0.jpg")
    _touch(tmp_path / "Cat" / "1.jpg")
    output = tmp_path / "metadata.jsonl"

    total_first, added_first = append_metadata(tmp_path, output, "a photo of a {class}")
    total_second, added_second = append_metadata(tmp_path, output, "a photo of a {class}")

    assert total_first == 2
    assert added_first == 2
    assert total_second == 2
    assert added_second == 0
    assert len(load_existing_metadata(output)) == 2


def test_overwrite_replaces_metadata(tmp_path: Path) -> None:
    _touch(tmp_path / "Cat" / "0.jpg")
    output = tmp_path / "metadata.jsonl"
    output.write_text(
        json.dumps({"file_name": "Cat/0.jpg", "text": "old caption"}) + "\n",
        encoding="utf-8",
    )

    _, added = append_metadata(tmp_path, output, "a photo of a {class}", overwrite=True)

    assert added == 1
    entries = list(output.read_text(encoding="utf-8").strip().splitlines())
    assert json.loads(entries[0])["text"] == "a photo of a cat"
