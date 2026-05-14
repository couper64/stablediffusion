# Checklist

* A CLI script that fine-tunes Stable Diffusion LoRA model and saves the best model in the `*.pt` format.
* A CLI script that generates images using the earlier trained best fine-tuned Stable Diffusion model.
* A CLI script that tests how good are the generated images by calculating an FID and Inception Score metrics, and saving the results in a JSON file.
* A CLI script that tests how much energy does an inference process consume using CodeCarbon, how long does it take to generate a single image using high precision clock and saving the results in a JSON file.
* CLI script that appends a standard classification task dataset with `metadata.jsonl` file that contains lines of file name and text fields per image, e.g. `{"file_name": "Cat/0.jpg", "text": "a photo of a cat"}`.

* Use `conda` to manage virtual environments.

* A global CLI script that defines a benchmarked pipeline: preprocess, train (optional), generate, evaluate. The pipeline's input is a folder with images grouped into classes. The output is a fine-tuned model, generated images, and metrics.