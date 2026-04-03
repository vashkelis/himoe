_base_ = 'mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'

custom_imports = dict(
    imports=[
        'projects.hi_moe.himoe_ffn',
        'projects.hi_moe.himoe_losses',
        'projects.hi_moe.himoe_hooks',
    ],
    allow_failed_imports=False,
)

num_experts = 16
num_scene_groups = 4
topk = 4
scene_topk = 2
use_scene_routing = True
use_instance_routing = True
use_shared_expert = True
apply_encoder_moe = True
apply_decoder_moe = True
max_epochs = 12
lambda_balance = 0.01
lambda_diversity = 0.001

moe_ffn_cfg = dict(
    type='HiMoEFFN',
    embed_dims=256,
    feedforward_channels=2048,
    num_experts=num_experts,
    num_scene_groups=num_scene_groups,
    topk=topk,
    scene_topk=scene_topk,
    ffn_drop=0.0,
    add_identity=True,
    use_shared_expert=use_shared_expert,
    use_scene_routing=use_scene_routing,
    use_instance_routing=use_instance_routing,
    act_cfg=dict(type='ReLU', inplace=True),
)

model = dict(
    encoder=dict(layer_cfg=dict(ffn_cfg=moe_ffn_cfg if apply_encoder_moe else None)),
    decoder=dict(layer_cfg=dict(ffn_cfg=moe_ffn_cfg if apply_decoder_moe else None)),
)

train_cfg = dict(max_epochs=max_epochs, val_interval=1)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2),
    logger=dict(type='LoggerHook', interval=50),
)
custom_hooks = [
    dict(type='SaveExpertUsageHook', filename_prefix='expert_usage', save_every_val=True),
    dict(type='HiMoEAuxLossHook', lambda_balance=lambda_balance, lambda_diversity=lambda_diversity),
]

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)
