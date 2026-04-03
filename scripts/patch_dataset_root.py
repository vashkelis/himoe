import argparse
from pathlib import Path


def patch_config(cfg_path: Path, dataset: str, data_root: str):
    text = cfg_path.read_text()
    if 'data_root =' in text:
        return
    if dataset == 'coco':
        patch = f"""

data_root = '{data_root.rstrip('/')}/'
train_dataloader = dict(dataset=dict(data_root=data_root, ann_file='annotations/instances_train2017.json', data_prefix=dict(img='train2017/')))
val_dataloader = dict(dataset=dict(data_root=data_root, ann_file='annotations/instances_val2017.json', data_prefix=dict(img='val2017/')))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator
"""
    else:
        patch = f"""

data_root = '{data_root.rstrip('/')}/'
"""
    cfg_path.write_text(text + patch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', default='coco')
    parser.add_argument('--data-root', required=True)
    args = parser.parse_args()
    patch_config(Path(args.config), args.dataset, args.data_root)


if __name__ == '__main__':
    main()
