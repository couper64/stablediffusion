# stablediffusion

CLI tools for Stable Diffusion LoRA workflows on class-folder image datasets:

- `stablediffusion-preprocess` — build `metadata.jsonl` captions from class subfolders.
- `stablediffusion-train` — fine-tune a base Stable Diffusion model with LoRA and save the **best** checkpoint as a single `*.pt` file.
- `stablediffusion-generate` — generate images with a trained LoRA (or the plain base model).
- `stablediffusion-eval` — score generated images with FID and Inception Score (JSON output).
- `stablediffusion-benchmark` — measure single-image inference latency and energy (JSON output).
- `stablediffusion-pipeline` — run preprocess → train → generate → evaluate end-to-end, with total time and energy for the full run.

This is the first piece of the larger job-queue service described in `CHECKLIST.md`; the same training/inference code will later be invoked by GPU workers.

## Install

Environments are managed with **conda**. The repo ships an `environment.yml` that creates an env named `stablediffusion` with PyTorch (CUDA 12.1) plus the Hugging Face stack:

```bash
conda env create -f environment.yml
conda activate stablediffusion
pip install -e .
```

On a CPU-only machine, replace `pytorch-cuda=12.1` with `cpuonly` in `environment.yml` before creating the env. Training Stable Diffusion on CPU is impractically slow; a CUDA-capable GPU is strongly recommended.

Updating after dependency changes:

```bash
conda env update -f environment.yml --prune
```

A standalone `requirements.txt` is also provided for pure-pip setups, but conda is the supported path.

## Dataset layout

Put your images in any folder (e.g. `data/my_set/`). Captions are optional and can be supplied in either of two ways:

1. **Sidecar text files** — `image.png` paired with `image.txt` containing the caption.
2. **`metadata.jsonl`** at the dataset root, one JSON object per line:

```json
{"file_name": "img001.png", "text": "a portrait of a corgi, studio light"}
{"file_name": "subdir/img002.jpg", "text": "a corgi on the beach at sunset"}
```

If neither is present, the value of `--default-caption` is used.

Supported extensions: `.jpg .jpeg .png .webp .bmp`.

## Train

```bash
stablediffusion-train \
    --data-dir data/my_set \
    --output checkpoints/best.pt \
    --base-model runwayml/stable-diffusion-v1-5 \
    --resolution 512 \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --epochs 30 \
    --learning-rate 1e-4 \
    --rank 8 \
    --val-split 0.1 \
    --mixed-precision fp16
```

The script:

1. Loads the base SD components (VAE, text encoder, UNet, scheduler) and **freezes everything**.
2. Attaches a LoRA adapter to the UNet attention layers (`to_q`, `to_k`, `to_v`, `to_out.0`).
3. Trains only the LoRA parameters with MSE on noise/`v_prediction` targets.
4. Validates each epoch on a held-out split (or falls back to training loss).
5. Whenever the tracked metric improves it overwrites `--output` with a single `*.pt` payload:

```python
{
  "lora_state_dict": {...},
  "lora_config": {"r": 8, "lora_alpha": 8, "target_modules": [...], "init_lora_weights": "gaussian"},
  "base_model": "runwayml/stable-diffusion-v1-5",
  "meta": {"epoch": ..., "train_loss": ..., "val_loss": ..., "resolution": 512}
}
```

Add `--save-every-epoch` if you also want per-epoch checkpoints alongside the best one.

## Generate

```bash
stablediffusion-generate \
    --lora checkpoints/best.pt \
    --prompt "a portrait of <subject>, cinematic lighting" \
    --negative-prompt "blurry, lowres" \
    --num-images 4 \
    --steps 30 --guidance-scale 7.5 \
    --seed 1234 \
    --output outputs/run1
```

The base model id is read from the checkpoint, so you only need to point at the `*.pt`. Pass `--base-model ...` to override. Omit `--lora` to use the plain base model.

## Project layout

```
src/stablediffusion/
    dataset.py       # ImageCaptionDataset
    util.py          # LoRA I/O, JSON helpers, upstream warning filters
    preprocess.py    # stablediffusion-preprocess — metadata.jsonl from class folders
    train.py         # stablediffusion-train
    generate.py      # stablediffusion-generate
    evaluate.py      # stablediffusion-eval — FID and Inception Score
    benchmark.py     # stablediffusion-benchmark — single-image latency and energy
    pipeline.py      # stablediffusion-pipeline — end-to-end workflow + total benchmark

tests/               # unit and integration tests
data/sample/         # bundled Cat/Dog smoke-test dataset (committed)
```

## Bundled sample dataset: `data/sample`

The repo includes `data/sample/` with 10 cat and 10 dog `.jpg` photos in class subfolders (`Cat/`, `Dog/`), plus a `metadata.jsonl` with per-class captions. Use it for quick end-to-end smoke tests:

**Preprocess** the sample dataset:

```
data/sample/
    Cat/0.jpg … Cat/9.jpg
    Dog/0.jpg … Dog/9.jpg
    metadata.jsonl
```

Regenerate captions after changing images with:

```bash
stablediffusion-preprocess --data-dir data/sample --overwrite
```

**Train** a LoRA adapter:

```bash
stablediffusion-train \
    --data-dir data/sample \
    --output output/model/sample.pt \
    --resolution 512 \
    --epochs 2 \
    --rank 8 \
    --val-split 0.1 \
    --mixed-precision fp16
```

**Generate** images from the checkpoint:

```bash
stablediffusion-generate \
    --lora output/model/sample.pt \
    --prompt "a fluffy ginger cat on a windowsill, soft natural light" \
    --num-images 10 --seed 1234 --output output/run1
```

**Evaluate** generated images against the reference set (FID and Inception Score, written to JSON):

```bash
stablediffusion-eval \
    --real-dir data/sample \
    --fake-dir output/run1 \
    --output output/metric/fid_is.json
```

Lower FID is better; higher Inception Score is better. Both metrics need a reasonably large image set to be meaningful — treat results on 20 images as a pipeline check only.

**Benchmark** single-image inference latency and energy (CodeCarbon + high-precision timer, written to JSON):

```bash
stablediffusion-benchmark \
    --prompt "a photo of a cat" \
    --lora output/model/sample.pt \
    --output output/metric/benchmark.json
```

GPU energy is measured via NVML; CPU/RAM figures are estimates (especially inside a VM without RAPL).
