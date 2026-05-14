"""End-to-end pipeline: preprocess → train → generate → evaluate.

Tracks total wall-clock time and energy for the full run via CodeCarbon.

Example::

    stablediffusion-pipeline \\
        --data-dir data/sample \\
        --output-dir output/pipeline_run

Skip training and reuse an existing LoRA::

    stablediffusion-pipeline \\
        --data-dir data/sample \\
        --output-dir output/pipeline_run \\
        --skip-train \\
        --lora output/model/sample.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from codecarbon import EmissionsTracker

from . import evaluate, generate, train
from .preprocess import append_metadata, caption_for_class, iter_classification_images
from .util import suppress_known_upstream_warnings, write_json_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full stablediffusion workflow on a class-folder dataset.")
    p.add_argument("--data-dir", required=True, help="Dataset root with one subfolder per class.")
    p.add_argument("--output-dir", required=True, help="Directory for model, images, and metrics.")
    p.add_argument("--caption-template", default="a photo of a {class}")
    p.add_argument("--skip-train", action="store_true", help="Skip LoRA training.")
    p.add_argument("--lora", default=None, help="Existing LoRA checkpoint; required with --skip-train.")
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    p.add_argument("--num-images", type=int, default=4, help="Generated images per class prompt.")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def list_class_names(data_dir: Path) -> List[str]:
    return sorted({rel.parts[0] for _, rel in iter_classification_images(data_dir)})


def prompts_for_classes(class_names: List[str], caption_template: str) -> List[str]:
    return [caption_for_class(name, caption_template) for name in class_names]


def _run_cli(module_main, argv: List[str]) -> None:
    previous = sys.argv
    sys.argv = argv
    try:
        module_main()
    finally:
        sys.argv = previous


def _execute_pipeline_steps(
    args: argparse.Namespace,
    *,
    data_dir: Path,
    metadata_path: Path,
    model_path: Path,
    generated_dir: Path,
    fid_path: Path,
    lora_path: Path,
    prompts: List[str],
) -> Path:
    print("=== [1/4] preprocess ===")
    total, added = append_metadata(
        data_dir=data_dir,
        output_path=metadata_path,
        caption_template=args.caption_template,
        overwrite=True,
    )
    print(f"metadata: {metadata_path} ({total} images, wrote {added} entries)")

    if args.skip_train:
        print("=== [2/4] train (skipped) ===")
        if not lora_path.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
        print(f"using checkpoint: {lora_path}")
    else:
        print("=== [2/4] train ===")
        _run_cli(
            train.main,
            [
                "stablediffusion-train",
                "--data-dir",
                str(data_dir),
                "--output",
                str(model_path),
                "--base-model",
                args.base_model,
                "--resolution",
                str(args.resolution),
                "--epochs",
                str(args.epochs),
                "--rank",
                str(args.rank),
                "--val-split",
                str(args.val_split),
                "--mixed-precision",
                args.mixed_precision,
                "--num-workers",
                str(args.num_workers),
            ],
        )
        lora_path = model_path

    print("=== [3/4] generate ===")
    generate_argv = [
        "stablediffusion-generate",
        "--lora",
        str(lora_path),
        "--output",
        str(generated_dir),
        "--num-images",
        str(args.num_images),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--height",
        str(args.resolution),
        "--width",
        str(args.resolution),
        "--seed",
        str(args.seed),
    ]
    for prompt in prompts:
        generate_argv.extend(["--prompt", prompt])
    _run_cli(generate.main, generate_argv)

    print("=== [4/4] evaluate ===")
    _run_cli(
        evaluate.main,
        [
            "stablediffusion-eval",
            "--real-dir",
            str(data_dir),
            "--fake-dir",
            str(generated_dir),
            "--output",
            str(fid_path),
        ],
    )
    return lora_path


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_names = list_class_names(data_dir)
    if not class_names:
        raise RuntimeError(
            f"No class-folder images found under {data_dir}. "
            "Expected layout like Cat/0.jpg with class names as subfolders."
        )

    if args.skip_train and not args.lora:
        raise ValueError("--lora is required when --skip-train is set")

    model_dir = output_dir / "model"
    generated_dir = output_dir / "generated"
    metrics_dir = output_dir / "metrics"
    model_path = model_dir / "lora.pt"
    fid_path = metrics_dir / "fid_is.json"
    pipeline_benchmark_path = metrics_dir / "pipeline_benchmark.json"
    metadata_path = data_dir / "metadata.jsonl"
    prompts = prompts_for_classes(class_names, args.caption_template)
    lora_path = Path(args.lora).resolve() if args.skip_train else model_path

    for path in (model_dir, generated_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(project_name="stablediffusion-pipeline", save_to_file=False)
    tracker.start()
    start = time.perf_counter()
    lora_path = _execute_pipeline_steps(
        args,
        data_dir=data_dir,
        metadata_path=metadata_path,
        model_path=model_path,
        generated_dir=generated_dir,
        fid_path=fid_path,
        lora_path=lora_path,
        prompts=prompts,
    )
    seconds_total = time.perf_counter() - start
    emissions_kg_co2 = tracker.stop()

    emissions: Dict[str, Any] = {}
    if tracker.final_emissions_data is not None:
        emissions = asdict(tracker.final_emissions_data)

    pipeline_benchmark = {
        "seconds_total": seconds_total,
        "emissions_kg_co2": emissions_kg_co2,
        "energy_kwh": emissions.get("energy_consumed"),
        "skipped_train": args.skip_train,
        "codecarbon": emissions,
    }
    write_json_results(pipeline_benchmark_path, pipeline_benchmark)

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "classes": class_names,
        "prompts": prompts,
        "metadata": str(metadata_path),
        "lora": str(lora_path),
        "generated_dir": str(generated_dir),
        "fid_is": str(fid_path),
        "pipeline_benchmark": str(pipeline_benchmark_path),
        "seconds_total": seconds_total,
        "emissions_kg_co2": emissions_kg_co2,
        "energy_kwh": emissions.get("energy_consumed"),
        "skipped_train": args.skip_train,
    }
    summary_path = output_dir / "pipeline.json"
    write_json_results(summary_path, summary)
    print(f"Pipeline complete in {seconds_total:.1f}s")
    if emissions_kg_co2 is not None:
        print(f"Emissions: {emissions_kg_co2 * 1000:.3f} g CO2eq")
    print(f"Summary: {summary_path}")
    print(f"Benchmark: {pipeline_benchmark_path}")
    return summary


def main() -> None:
    suppress_known_upstream_warnings()
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
