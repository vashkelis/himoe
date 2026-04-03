from typing import Dict

from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class HiMoEAuxLossHook(Hook):
    """Inject auxiliary MoE losses into the runner outputs.

    This starter hook expects the model's train_step to return a dict with key
    'loss'. Since MMDetection models don't expose FFN aux losses by default,
    this hook logs aggregate losses and can optionally reset stats each iter.
    In a production implementation, wire these losses into the detector's loss
    dict directly.
    """
    priority = 'LOW'

    def __init__(self, lambda_balance: float = 0.01, lambda_diversity: float = 0.001, reset_stats_each_iter: bool = True):
        self.lambda_balance = lambda_balance
        self.lambda_diversity = lambda_diversity
        self.reset_stats_each_iter = reset_stats_each_iter

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs: Dict = None):
        # Best-effort logging only.
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        bal = 0.0
        div = 0.0
        count = 0
        for _, module in model.named_modules():
            if hasattr(module, 'aux_losses'):
                losses = module.aux_losses()
                bal += float(losses['loss_balance'].detach().cpu())
                div += float(losses['loss_diversity'].detach().cpu())
                count += 1
                if self.reset_stats_each_iter and hasattr(module, 'reset_online_stats'):
                    module.reset_online_stats()
        if count > 0:
            runner.message_hub.update_scalar('train/himoe_loss_balance', bal / count)
            runner.message_hub.update_scalar('train/himoe_loss_diversity', div / count)
