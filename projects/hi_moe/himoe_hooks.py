from pathlib import Path

import numpy as np
from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class SaveExpertUsageHook(Hook):
    priority = 'LOW'

    def __init__(self, filename_prefix='expert_usage', save_every_val=True):
        self.filename_prefix = filename_prefix
        self.save_every_val = save_every_val

    def _collect(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        rows = []
        for name, module in model.named_modules():
            if hasattr(module, 'expert_usage_counts'):
                steps = float(module.routing_steps.detach().cpu().item()) if hasattr(module, 'routing_steps') else 0.0
                entropy = float((module.routing_entropy_sum / max(steps, 1.0)).detach().cpu().item()) if steps else 0.0
                rows.append(dict(
                    name=name,
                    expert_usage_counts=module.expert_usage_counts.detach().cpu().numpy(),
                    expert_usage_ema=module.expert_usage_ema.detach().cpu().numpy(),
                    scene_usage_counts=module.scene_usage_counts.detach().cpu().numpy(),
                    total_routed_tokens=float(module.total_routed_tokens.detach().cpu().item()),
                    avg_routing_entropy=entropy,
                ))
        return rows

    def _save(self, runner, suffix):
        out_dir = Path(runner.work_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f'{self.filename_prefix}_{suffix}.npy'
        np.save(path, self._collect(runner), allow_pickle=True)
        runner.logger.info(f'Saved expert usage to {path}')

    def after_val_epoch(self, runner, metrics=None):
        if self.save_every_val:
            self._save(runner, f'val_epoch_{runner.epoch + 1}')

    def after_test_epoch(self, runner, metrics=None):
        self._save(runner, 'test')
