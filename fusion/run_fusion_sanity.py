# run_fusion_sanity.py
# 그냥 multi_fusion.py 잘 되는지 확인하는 포워딩 코드 
import torch
from multi_fusion import FleximodalFuseMoE, FlexiConfig, ProjectionCfg

modalities = ["timeseries","image","text"]
proj_cfgs = {
  "timeseries": ProjectionCfg(in_dim=256,  max_len=256),   # 예시
  "image":      ProjectionCfg(in_dim=1024, max_len=64),    # 예시
  "text":       ProjectionCfg(in_dim=768,  max_len=128),   # 예시
}
cfg = FlexiConfig(d_model=256, n_heads=8, n_layers=2, n_experts=4,
                  latent_len=64, num_classes=2)

model = FleximodalFuseMoE(modalities, proj_cfgs, cfg).cuda().eval()
B = 8
feats = {
  "timeseries": torch.randn(B, 256, 256).cuda(),  # [B, T_ts, d_ts]
  "image":      torch.randn(B, 64, 1024).cuda(),  # [B, T_img, d_img]
  "text":       torch.randn(B, 128, 768).cuda(),  # [B, T_txt, d_txt]
}
out = model(feats)
print("logits:", out["logits"].shape, "cls:", out["cls"].shape)  # [B,2], [B,256]
