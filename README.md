# HI-MoE: Hierarchical Instance-Conditioned Mixture-of-Experts for Object Detection

Official Google Colab experiments for the paper:

**HI-MoE: Hierarchical Instance-Conditioned Mixture-of-Experts for Object Detection**

This repository provides **Google Colab-ready code** to reproduce experiments, ablations, and analysis for HI-MoE, a hierarchical mixture-of-experts architecture for object detection.

## Overview

Mixture-of-Experts (MoE) architectures scale model capacity by routing inputs to specialized subnetworks. While MoE has been highly successful in language models, most object detectors remain dense.

HI-MoE introduces:

- scene-level routing
- instance-level routing
- hierarchical expert selection
- instance-conditioned sparse computation

The method is designed for heterogeneous detection scenes containing:

- small objects
- large objects
- occluded instances
- crowded scenes
- long-tail categories

HI-MoE builds on detection transformers such as **DINO** and replaces selected FFN layers with **hierarchical MoE modules**.

## Key Idea

Prior vision MoE methods mostly route at the image or token level. HI-MoE instead routes **object queries**.

The routing process has two stages:

1. **Scene routing** chooses a scene-conditioned pool of experts.
2. **Instance routing** selects top-k experts for each object query.

This allows specialization inside a single image, rather than only across datasets or tasks.

## Repository Structure

```text
configs/
    generated/
    himoe_config_template.py

projects/
    hi_moe/
        himoe_ffn.py
        routing.py
        hooks.py
        plot_expert_usage.py

scripts/
    setup_colab.sh
    generate_configs.py
    patch_dataset_root.py
    run_ablation_grid.py

HI_MoE_Colab.ipynb
README.md
CONTRIBUTING.md
requirements.txt
```

## Recommended Google Drive Layout

```text
MyDrive/
└── HI_MOE/
    ├── datasets/
    │   ├── coco/
    │   ├── lvis/
    │   └── objects365/
    ├── outputs/
    │   ├── work_dirs/
    │   ├── logs/
    │   └── figures/
    ├── cache/
    └── notebooks/
```

## Quick Start in Google Colab

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Clone the repository

```bash
git clone <YOUR_GITHUB_REPO_URL> /content/hi_moe
cd /content/hi_moe
```

### 3. Install dependencies

```bash
bash scripts/setup_colab.sh
```

### 4. Patch dataset roots

```bash
python scripts/patch_dataset_root.py \
  --dataset_root /content/drive/MyDrive/HI_MOE/datasets \
  --input_dir configs \
  --output_dir configs/generated
```

### 5. Generate experiment configs

```bash
python scripts/generate_configs.py
```

### 6. Run a pilot experiment

```bash
python tools/train.py configs/generated/himoe_full.py
```

### 7. Run the ablation grid

```bash
python scripts/run_ablation_grid.py
```

## Datasets

This codebase is organized around experiments on:

- **COCO**
- **LVIS**
- **Objects365**

Expected COCO layout:

```text
datasets/coco/
    train2017/
    val2017/
    annotations/
        instances_train2017.json
        instances_val2017.json
```

Expected LVIS layout:

```text
datasets/lvis/
    annotations/
    train2017/
    val2017/
```

Expected Objects365 layout depends on the specific preprocessing pipeline used in your experiments.

## Experiments Included

### Main model

- full HI-MoE

### Ablations

- token-level MoE baseline
- instance-only routing
- scene-only routing
- top-k routing ablations
- number-of-experts ablations
- MoE placement ablations

## Example Results

| Method | AP | APs |
|---|---:|---:|
| DINO | 51.3 | 32.1 |
| HI-MoE | 53.0 | 35.4 |

## Expert Analysis and Visualization

Generate the expert usage heatmap with:

```bash
python projects/hi_moe/plot_expert_usage.py
```

Expected outputs:

```text
outputs/figures/expert_usage.png
outputs/figures/routing_entropy.png
```

A typical expert usage visualization uses:

- rows = decoder layers
- columns = experts
- values = routing frequency

This helps show whether deeper layers become more specialized.

## Reproducibility Notes

For a stable experimental workflow:

- keep datasets and outputs in Google Drive
- keep code in GitHub
- clone fresh into `/content/hi_moe` each Colab session
- run one smoke test before running the full ablation grid

Free Colab is usually sufficient for smoke tests and small runs. Full paper-scale experiments may require Colab Pro or dedicated GPUs.

## Built On

This project is built around:

- PyTorch
- MMDetection
- DINO / DETR-style detectors
- COCO API

## Citation

```bibtex
@article{himoe2025,
  title={HI-MoE: Hierarchical Instance-Conditioned Mixture-of-Experts for Object Detection},
  author={Vashkelis, Vadim and Trukhina, Natalia},
  year={2025}
}
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Choose and add a license before public release. MIT is a common choice for research code.

## Status

Research prototype. Experimental code under active development.
