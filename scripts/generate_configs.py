from pathlib import Path

TEMPLATE = Path('/mnt/data/himoe_experiments/configs/himoe_config_template.py').read_text()
OUT = Path('/mnt/data/himoe_experiments/configs/generated')
OUT.mkdir(parents=True, exist_ok=True)

variants = {
    'himoe_full_12e.py': dict(num_experts=16, num_scene_groups=4, topk=2, scene_topk=2, use_scene_routing='True', use_instance_routing='True', use_shared_expert='True', apply_encoder_moe='True', apply_decoder_moe='True', max_epochs=12, lambda_balance=0.01, lambda_diversity=0.001),
    'himoe_token_moe_12e.py': dict(num_experts=16, num_scene_groups=1, topk=2, scene_topk=1, use_scene_routing='False', use_instance_routing='True', use_shared_expert='True', apply_encoder_moe='True', apply_decoder_moe='True', max_epochs=12, lambda_balance=0.01, lambda_diversity=0.001),
    'himoe_instance_only_12e.py': dict(num_experts=16, num_scene_groups=1, topk=2, scene_topk=1, use_scene_routing='False', use_instance_routing='True', use_shared_expert='True', apply_encoder_moe='False', apply_decoder_moe='True', max_epochs=12, lambda_balance=0.01, lambda_diversity=0.001),
    'himoe_scene_only_12e.py': dict(num_experts=16, num_scene_groups=4, topk=1, scene_topk=2, use_scene_routing='True', use_instance_routing='False', use_shared_expert='True', apply_encoder_moe='False', apply_decoder_moe='True', max_epochs=12, lambda_balance=0.01, lambda_diversity=0.001),
    'himoe_k1_12e.py': dict(num_experts=16, num_scene_groups=4, topk=1, scene_topk=2, use_scene_routing='True', use_instance_routing='True', use_shared_expert='True', apply_encoder_moe='True', apply_decoder_moe='True', max_epochs=12, lambda_balance=0.01, lambda_diversity=0.001),
    'himoe_k4_12e.py': dict(num_experts=16, num_scene_groups=4, topk=4, scene_topk=2, use_scene_routing='True', use_instance_routing='True', use_shared_expert='True', apply_encoder_moe='True', apply_decoder_moe='True', max_epochs=12, lambda_balance=0.01, lambda_diversity=0.001),
    'himoe_e4_12e.py': dict(num_experts=4, num_scene_groups=2, topk=2, scene_topk=1, use_scene_routing='True', use_instance_routing='True', use_shared_expert='True', apply_encoder_moe='True', apply_decoder_moe='True', max_epochs=12, lambda_balance=0.01, lambda_diversity=0.001),
    'himoe_e8_12e.py': dict(num_experts=8, num_scene_groups=4, topk=2, scene_topk=2, use_scene_routing='True', use_instance_routing='True', use_shared_expert='True', apply_encoder_moe='True', apply_decoder_moe='True', max_epochs=12, lambda_balance=0.01, lambda_diversity=0.001),
}

for name, vals in variants.items():
    text = TEMPLATE
    for k, v in vals.items():
        text = text.replace('{{ ' + k + ' }}', str(v))
    (OUT / name).write_text(text)
    print('wrote', OUT / name)
