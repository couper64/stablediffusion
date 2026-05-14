# Checklist

* A CLI script that fine-tunes Stable Diffusion LoRA model and saves the best model in the `*.pt` format.
* A CLI script that generates images using the earlier trained best fine-tuned Stable Diffusion model.
* A CLI script that tests how good are the generated images by calculating an FID and Inception Score metrics, and saving the results in a JSON file.
* A CLI script that tests how much energy does an inference process consume using CodeCarbon, how long does it take to generate a single image using high precision clock and saving the results in a JSON file.

* Use `conda` to manage virtual environments.