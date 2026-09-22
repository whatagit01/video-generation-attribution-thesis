# Video Generation & Method Attribution: GAN vs. Diffusion Under Resource Constraints

MSc Computer Science thesis (QMUL) comparing GAN and diffusion video generation on a single GPU, plus a hybrid CNN-Transformer model to detect which architecture generated a given video (real / GAN / diffusion).

## What this is

Trained MoStGAN-V (GAN) and Latte (diffusion) on a single A40 GPU, using a small, publicly-sourced dataset of political speech and combat footage. Then built a lightweight CNN + TimeSformer-based classifier to test whether generated video can be attributed back to its source model, not just flagged as fake.

## Results

| | MoStGAN-V | Latte |
|---|---|---|
| FVD (actions) | 1075.00 | 3099.35 |
| FVD (speech) | 262.70 | 2300.41 |
| LPIPS (actions) | 0.70 | 0.89 |
| LPIPS (speech) | 0.58 | 0.79 |

- MoStGAN-V beat Latte by 65–89% on FVD, and plateaued within its training budget — Latte was still improving when I stopped training
- Attribution classifier hit 92.9–95.4% test accuracy, with diffusion samples almost perfectly identified
- Removing MoStGAN-V's best (speech) samples didn't improve accuracy like I expected — suggests the classifier is picking up on real, attributable artefacts rather than just exploiting quality differences

## Structure

- `attribution_model/` — CNN-Transformer architecture, training + evaluation scripts
- `preprocessing/` — video preprocessing (cropping, border removal, segmentation)
- `evaluation/` — FVD/LPIPS computation
- `environment_setup/` — conda environment setup notebooks
- `results/` — FVD/LPIPS results across generation checkpoints

## Note

Checkpoints, generated videos, and raw data aren't included here due to size — this repo has the code, setup, and results.
