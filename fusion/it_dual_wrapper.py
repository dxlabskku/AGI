# it_dual_wrapper.py
import torch
from it_baseline_train import ITBackbone, CFG, l2_normalize

class ITDualWrapper(torch.nn.Module):
    def __init__(self, ckpt_path: str, dim: int = 256, device: str = "cuda"):
        super().__init__()
        cfg = CFG(dim=dim, device=device)
        self.m = ITBackbone(cfg).to(device).eval()
        if ckpt_path:
            from it_baseline_train import load_ckpt
            load_ckpt(ckpt_path, self.m)
        for p in self.m.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def encode_image_only(self, images):
        # images: [B,3,224,224] (CLIP space)
        z_img = self.m.encode_image(images.to(self.device))          # [B,256] (L2 normed)
        return z_img[:, None, :]                                     # [B,1,256]

    @torch.no_grad()
    def encode_text_only(self, texts: list[str], use_fused: bool = True):
        z_c = self.m.encode_text_clip(texts)                         # [B,256]
        z_e = self.m.encode_text_e5(texts)                           # [B,256]
        if use_fused:
            # Gated fuse (모델 내부 fuser와 동일)
            z_txt, _ = self.m.fuser(z_c, z_e)                        # [B,256]
        else:
            z_txt = l2_normalize((z_c + z_e) / 2)                    # 간단 평균
        return z_txt[:, None, :]                                     # [B,1,256]

    @torch.no_grad()
    def forward(self, images=None, texts: list[str] | None = None):
        feats = {}
        if images is not None:
            feats["image"] = self.encode_image_only(images)          # [B,1,256]
        if texts is not None and len(texts) > 0:
            feats["text"]  = self.encode_text_only(texts)            # [B,1,256]
        return feats  # -> multi_fusion으로 바로 투입
