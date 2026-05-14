"""CLI to score generated images with FID and Inception Score.

Example::

    stablediff-eval \\
        --real-dir data/sample \\
        --fake-dir outputs/run1 \\
        --output metrics/fid_is.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_fidelity import calculate_metrics

from .util import suppress_known_upstream_warnings, write_json_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute FID and Inception Score for generated images.")
    p.add_argument("--real-dir", required=True, help="Directory with reference (real) images.")
    p.add_argument("--fake-dir", required=True, help="Directory with generated images to score.")
    p.add_argument("--output", required=True, help="Path to write JSON results.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for metric computation (default: auto).",
    )
    return p.parse_args()


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def main() -> None:
    suppress_known_upstream_warnings()
    args = parse_args()
    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    if not real_dir.is_dir():
        raise FileNotFoundError(f"Real image directory not found: {real_dir}")
    if not fake_dir.is_dir():
        raise FileNotFoundError(f"Generated image directory not found: {fake_dir}")

    device = _resolve_device(args.device)
    print(f"Computing FID and Inception Score (device={device})")
    print(f"  real: {real_dir}")
    print(f"  fake: {fake_dir}")

    metrics = calculate_metrics(
        input1=str(real_dir),
        input2=str(fake_dir),
        cuda=device == "cuda",
        batch_size=args.batch_size,
        save_cpu_ram=True,
        samples_find_deep=True,
        fid=True,
        isc=True,
    )

    results = {
        "real_dir": str(real_dir.resolve()),
        "fake_dir": str(fake_dir.resolve()),
        "device": device,
        "batch_size": args.batch_size,
        "fid": metrics["frechet_inception_distance"],
        "inception_score_mean": metrics["inception_score_mean"],
        "inception_score_std": metrics["inception_score_std"],
    }
    output_path = Path(args.output)
    write_json_results(output_path, results)
    print(
        f"FID={results['fid']:.4f} "
        f"IS={results['inception_score_mean']:.4f} "
        f"(±{results['inception_score_std']:.4f})"
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
