# HI-MoE Experiment Pack for Google Colab

This pack gives you a **single-repo starting point** for the HI-MoE paper experiments on top of **MMDetection + DINO**.

Included:
- `projects/hi_moe/himoe_ffn.py`: hierarchical scene-to-instance MoE FFN
- `projects/hi_moe/himoe_hooks.py`: save per-layer expert usage for paper figures
- `projects/hi_moe/himoe_losses.py`: auxiliary loss logging hook
- `configs/generated/*.py`: COCO 12e ablation configs
- `scripts/setup_colab.sh`: Colab install script
- `scripts/run_ablation_grid.py`: runs the paper ablation grid
- `projects/hi_moe/plot_expert_usage.py`: builds `expert_usage.png`

## What this can do now
- Run a DINO-based COCO pilot on Colab
- Run the ablation grid from the paper: token-level, instance-only, scene-only, full, top-k variants, expert-count variants
- Save `expert_usage_*.npy`
- Produce `expert_usage.png`

## What likely needs stronger hardware
- 24e / 50e COCO runs
- Objects365 pretraining
- LVIS full fine-tuning

## Colab workflow
1. Upload the `himoe_experiments` folder to `/content/`.
2. Run `scripts/setup_colab.sh`.
3. Patch dataset roots with `patch_dataset_root.py`.
4. Train one config or run the ablation grid.
5. Test and call `plot_expert_usage.py`.

## Caveat
This is a **best-effort research scaffold**. Exact FFN injection points can depend on the MMDetection / DINO config version you install, so you may need minor config adaptation if upstream internals change.
