#!/usr/bin/env bash
set -euxo pipefail
pip -q install -U openmim
mim install -q 'mmengine>=0.10.0'
mim install -q 'mmcv>=2.0.0'
pip -q install 'mmdet>=3.3.0' pycocotools lvis matplotlib seaborn pandas
if [ ! -d /content/mmdetection ]; then
  git clone https://github.com/open-mmlab/mmdetection.git /content/mmdetection
fi
cd /content/mmdetection
pip -q install -e .
mkdir -p /content/mmdetection/projects/hi_moe
cp -r /content/himoe_experiments/projects/hi_moe/* /content/mmdetection/projects/hi_moe/
cp /content/himoe_experiments/configs/generated/*.py /content/mmdetection/projects/hi_moe/
