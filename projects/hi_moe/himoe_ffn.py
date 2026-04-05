# Patch inspect.getsource to handle PyTorch builtins without source
# Must be applied BEFORE importing mmengine/torch that triggers the issue
import inspect
_original_getsource = inspect.getsource
def _patched_getsource(obj):
    try:
        return _original_getsource(obj)
    except (OSError, TypeError):
        return "def _torch_builtin_(): pass"
inspect.getsource = _patched_getsource

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class HiMoEFFN(BaseModule):
    """Hierarchical instance-conditioned Mixture-of-Experts FFN.

    Designed to replace FFN modules in DETR/DINO transformer layers.
    Input shape can be [N, B, C] or [B, N, C].
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 2048,
        num_experts: int = 16,
        num_scene_groups: int = 4,
        topk: int = 2,
        scene_topk: int = 2,
        ffn_drop: float = 0.0,
        act_cfg: Optional[dict] = None,
        add_identity: bool = True,
        use_shared_expert: bool = True,
        use_scene_routing: bool = True,
        use_instance_routing: bool = True,
        usage_momentum: float = 0.95,
        tau: float = 1.0,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_experts = num_experts
        self.num_scene_groups = num_scene_groups
        self.topk = topk
        self.scene_topk = scene_topk
        self.add_identity = add_identity
        self.use_shared_expert = use_shared_expert
        self.use_scene_routing = use_scene_routing
        self.use_instance_routing = use_instance_routing
        self.usage_momentum = usage_momentum
        self.tau = tau

        act_type = (act_cfg or {}).get('type', 'ReLU')
        if act_type == 'ReLU':
            def make_act():
                return nn.ReLU(inplace=(act_cfg or {}).get('inplace', True))
        elif act_type == 'GELU':
            def make_act():
                return nn.GELU()
        else:
            raise ValueError(f'Unsupported act_cfg: {act_cfg}')

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                make_act(),
                nn.Dropout(ffn_drop),
                nn.Linear(feedforward_channels, embed_dims),
                nn.Dropout(ffn_drop),
            )
            for _ in range(num_experts)
        ])
        self.shared_expert = None
        if use_shared_expert:
            self.shared_expert = nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                make_act(),
                nn.Dropout(ffn_drop),
                nn.Linear(feedforward_channels, embed_dims),
                nn.Dropout(ffn_drop),
            )

        self.scene_router = nn.Linear(embed_dims, num_scene_groups)
        self.instance_router = nn.Linear(embed_dims + num_scene_groups, num_experts)
        expert_to_group = torch.arange(num_experts) % max(num_scene_groups, 1)
        self.register_buffer('expert_to_group', expert_to_group, persistent=False)

        self.register_buffer('expert_usage_counts', torch.zeros(num_experts), persistent=True)
        self.register_buffer('expert_usage_ema', torch.zeros(num_experts), persistent=True)
        self.register_buffer('scene_usage_counts', torch.zeros(max(num_scene_groups, 1)), persistent=True)
        self.register_buffer('routing_entropy_sum', torch.tensor(0.0), persistent=True)
        self.register_buffer('routing_steps', torch.tensor(0.0), persistent=True)
        self.register_buffer('total_routed_tokens', torch.tensor(0.0), persistent=True)

    def _normalize_layout(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f'Expected 3D tensor, got {tuple(x.shape)}')
        transposed = False
        # DETR internals usually use [N, B, C]
        if x.shape[0] > x.shape[1]:
            x = x.transpose(0, 1)
            transposed = True
        return x, transposed

    def _scene_mask(self, scene_probs: torch.Tensor) -> torch.Tensor:
        # scene_probs [B, G] -> mask [B, E]
        if not self.use_scene_routing:
            return torch.ones(scene_probs.shape[0], self.num_experts, device=scene_probs.device, dtype=torch.bool)
        k = min(self.scene_topk, self.num_scene_groups)
        top_groups = scene_probs.topk(k=k, dim=-1).indices
        active_groups_mask = torch.zeros_like(scene_probs, dtype=torch.bool)
        active_groups_mask.scatter_(1, top_groups, True)
        return active_groups_mask[:, self.expert_to_group]

    def _update_usage(self, selected_experts: torch.Tensor, scene_probs: torch.Tensor, routing_probs: torch.Tensor):
        with torch.no_grad():
            flat = selected_experts.reshape(-1)
            counts = torch.bincount(flat, minlength=self.num_experts).float().to(self.expert_usage_counts.device)
            self.expert_usage_counts += counts
            denom = counts.sum().clamp_min(1.0)
            freqs = counts / denom
            self.expert_usage_ema.mul_(self.usage_momentum).add_((1 - self.usage_momentum) * freqs)
            self.total_routed_tokens += float(flat.numel())
            if scene_probs.numel() > 0:
                self.scene_usage_counts += scene_probs.sum(dim=0)
            p = routing_probs.clamp_min(1e-8)
            ent = -(p * p.log()).sum(dim=-1).mean()
            self.routing_entropy_sum += ent.detach()
            self.routing_steps += 1.0

    def aux_losses(self) -> Dict[str, torch.Tensor]:
        # Compute on cumulative counts; hook resets per-iteration in training.
        if self.expert_usage_counts.sum() <= 0:
            zero = self.expert_usage_counts.sum() * 0.0
            return {'loss_balance': zero, 'loss_diversity': zero}
        freqs = self.expert_usage_counts / self.expert_usage_counts.sum().clamp_min(1.0)
        target = torch.full_like(freqs, 1.0 / self.num_experts)
        loss_balance = ((freqs - target) ** 2).sum()
        p = freqs.clamp_min(1e-8)
        entropy = -(p * p.log()).sum()
        max_entropy = math.log(self.num_experts)
        loss_diversity = (max_entropy - entropy) / max_entropy
        return {'loss_balance': loss_balance, 'loss_diversity': loss_diversity}

    def reset_online_stats(self):
        self.expert_usage_counts.zero_()
        self.scene_usage_counts.zero_()
        self.total_routed_tokens.zero_()
        self.routing_entropy_sum.zero_()
        self.routing_steps.zero_()

    def forward(self, x: torch.Tensor, identity: Optional[torch.Tensor] = None):
        x, transposed = self._normalize_layout(x)  # [B, N, C]
        b, n, c = x.shape
        identity_ = x if identity is None else identity.transpose(0, 1) if (transposed and identity.dim() == 3) else identity
        if identity_ is None:
            identity_ = x

        pooled = x.mean(dim=1)
        if self.num_scene_groups > 0:
            scene_logits = self.scene_router(pooled) / self.tau
            scene_probs = F.softmax(scene_logits, dim=-1)
        else:
            scene_probs = x.new_zeros((b, 1))
        expert_mask = self._scene_mask(scene_probs)

        if self.use_instance_routing:
            scene_ctx = scene_probs.unsqueeze(1).expand(-1, n, -1)
            router_inp = torch.cat([x, scene_ctx], dim=-1)
            inst_logits = self.instance_router(router_inp) / self.tau
        else:
            inst_logits = x.new_zeros((b, n, self.num_experts))

        inst_logits = inst_logits.masked_fill(~expert_mask.unsqueeze(1), float('-inf'))
        k = min(self.topk, self.num_experts)
        topk_scores, topk_idx = inst_logits.topk(k=k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)
        routing_probs = F.softmax(inst_logits, dim=-1)

        x_flat = x.reshape(b * n, c)
        expert_outs = torch.stack([expert(x_flat).reshape(b, n, c) for expert in self.experts], dim=2)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, c)
        selected = torch.gather(expert_outs, 2, gather_idx)
        out = (selected * topk_weights.unsqueeze(-1)).sum(dim=2)
        if self.shared_expert is not None:
            out = out + self.shared_expert(x_flat).reshape(b, n, c)
        if self.add_identity:
            out = out + identity_

        self._update_usage(topk_idx.detach(), scene_probs.detach(), routing_probs.detach())
        if transposed:
            out = out.transpose(0, 1)
        return out
