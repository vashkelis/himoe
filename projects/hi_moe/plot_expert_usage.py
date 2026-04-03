import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path):
    arr = np.load(path, allow_pickle=True)
    return list(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--usage-npy', type=str, required=True)
    parser.add_argument('--out', type=str, default='expert_usage.png')
    parser.add_argument('--title', type=str, default='Expert utilization across decoder / encoder MoE layers')
    parser.add_argument('--use-ema', action='store_true')
    args = parser.parse_args()

    records = _load(args.usage_npy)
    labels = [r['name'] for r in records]
    usage = np.stack([
        r['expert_usage_ema'] if args.use_ema else np.asarray(r['expert_usage_counts']) / max(np.asarray(r['expert_usage_counts']).sum(), 1e-9)
        for r in records
    ], axis=0)
    entropy = np.asarray([r.get('avg_routing_entropy', 0.0) for r in records])

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.35 * len(labels))), constrained_layout=True)
    im = axes[0].imshow(usage, aspect='auto')
    axes[0].set_title('Usage heatmap')
    axes[0].set_xlabel('Experts')
    axes[0].set_ylabel('MoE layers')
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].plot(entropy, np.arange(len(labels)), marker='o')
    axes[1].invert_yaxis()
    axes[1].set_title('Avg routing entropy')
    axes[1].set_xlabel('Entropy')
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels([])
    fig.suptitle(args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
