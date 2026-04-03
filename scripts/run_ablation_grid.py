import argparse
import os
import subprocess
from pathlib import Path

VARIANTS = [
    'himoe_full_12e.py',
    'himoe_token_moe_12e.py',
    'himoe_instance_only_12e.py',
    'himoe_scene_only_12e.py',
    'himoe_k1_12e.py',
    'himoe_k4_12e.py',
    'himoe_e4_12e.py',
    'himoe_e8_12e.py',
]


def run(cmd, cwd=None):
    print('RUN', ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mmdet-root', default='/content/mmdetection')
    parser.add_argument('--variant-dir', default='/content/mmdetection/projects/hi_moe')
    parser.add_argument('--work-base', default='/content/drive/MyDrive/himoe_runs')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    Path(args.work_base).mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        cfg = f'{args.variant_dir}/{variant}'
        work_dir = f'{args.work_base}/{Path(variant).stem}'
        cmd = ['python', 'tools/train.py', cfg, '--work-dir', work_dir]
        if args.amp:
            cmd.append('--amp')
        run(cmd, cwd=args.mmdet_root)


if __name__ == '__main__':
    main()
