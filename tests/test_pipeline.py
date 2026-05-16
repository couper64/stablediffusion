from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import pytest

from stablediffusion.pipeline import list_class_names, prompts_for_classes, run_pipeline


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


@contextmanager
def _fake_track_emissions(_project_name: str):
    yield {
        "seconds": 0.0,
        "energy_kwh": 0.001,
        "emissions_kg_co2": 0.001,
        "energy_breakdown": None,
        "codecarbon": {},
    }


def test_list_class_names_and_prompts(tmp_path: Path) -> None:
    _touch(tmp_path / "Cat" / "0.jpg")
    _touch(tmp_path / "Dog" / "1.jpg")

    names = list_class_names(tmp_path)
    prompts = prompts_for_classes(names, "a photo of a {class}")

    assert names == ["Cat", "Dog"]
    assert prompts == ["a photo of a cat", "a photo of a dog"]


def test_run_pipeline_steps_and_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch(tmp_path / "Cat" / "0.jpg")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for src in (tmp_path / "Cat").iterdir():
        (data_dir / "Cat").mkdir(parents=True, exist_ok=True)
        (data_dir / "Cat" / src.name).write_bytes(src.read_bytes())

    called: list[str] = []

    def fake_cli(name: str):
        def _main() -> None:
            called.append(name)

        return _main

    monkeypatch.setattr("stablediffusion.pipeline.track_emissions", _fake_track_emissions)
    monkeypatch.setattr("stablediffusion.pipeline.train.main", fake_cli("train"))
    monkeypatch.setattr("stablediffusion.pipeline.generate.main", fake_cli("generate"))
    monkeypatch.setattr("stablediffusion.pipeline.evaluate.main", fake_cli("evaluate"))

    output_dir = tmp_path / "out"
    summary = run_pipeline(
        argparse.Namespace(
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            caption_template="a photo of a {class}",
            skip_train=False,
            lora=None,
            base_model="runwayml/stable-diffusion-v1-5",
            epochs=1,
            rank=4,
            resolution=256,
            val_split=0.0,
            mixed_precision="fp16",
            num_images=1,
            steps=5,
            guidance_scale=7.5,
            seed=1,
            num_workers=0,
            disable_safety_checker=False,
        )
    )

    assert (data_dir / "metadata.jsonl").is_file()
    assert called == ["train", "generate", "evaluate"]
    assert summary["classes"] == ["Cat"]
    assert (output_dir / "pipeline.json").is_file()
    assert (output_dir / "metrics" / "pipeline_benchmark.json").is_file()
    assert summary["seconds_total"] >= 0
    assert summary["emissions_kg_co2"] == 0.001


def test_skip_train_requires_lora(tmp_path: Path) -> None:
    _touch(tmp_path / "Cat" / "0.jpg")
    args = argparse.Namespace(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        caption_template="a photo of a {class}",
        skip_train=True,
        lora=None,
        base_model="runwayml/stable-diffusion-v1-5",
        epochs=1,
        rank=4,
        resolution=256,
        val_split=0.0,
        mixed_precision="fp16",
        num_images=1,
        steps=5,
        guidance_scale=7.5,
        seed=1,
        num_workers=0,
        disable_safety_checker=False,
    )
    with pytest.raises(ValueError, match="--lora is required"):
        run_pipeline(args)
