

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------

def _init_weights(module: nn.Module):
    if isinstance(module, (nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


class RMSNorm(nn.Module):
    """RMSNorm for stability (no mean subtraction).
    """
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.scale * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))


# -----------------------------
# Projection bank (per modality)
# -----------------------------

@dataclass
class ProjectionCfg:
    in_dim: int
    max_len: int = 512  # max tokens per modality sequence


class ProjectionBank(nn.Module):
    """Per-modality projection + LayerNorm + Dropout.

    Args:
        common_dim: shared model dimension D
        modalities: list of modality names (keys)
        proj_cfgs: dict[modality] -> ProjectionCfg (in_dim, max_len)
        p_drop: dropout prob after projection
    """
    def __init__(self, common_dim: int, modalities: List[str], proj_cfgs: Dict[str, ProjectionCfg], p_drop: float = 0.1):
        super().__init__()
        self.D = common_dim
        self.modalities = modalities
        self.proj = nn.ModuleDict({m: nn.Linear(proj_cfgs[m].in_dim, common_dim) for m in modalities})
        self.ln = nn.ModuleDict({m: nn.LayerNorm(common_dim) for m in modalities})
        self.drop = nn.Dropout(p_drop)
        self.register_buffer("max_len_tensor", torch.tensor([max(proj_cfgs[m].max_len for m in modalities)]), persistent=False)
        self.apply(_init_weights)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """feats[m]: [B, T_m, d_in_m] -> tokens[m]: [B, T_m, D]
        Only modalities present in feats are processed.
        """
        out = {}
        for m, x in feats.items():
            z = self.proj[m](x)
            z = self.ln[m](z)
            z = self.drop(z)
            out[m] = z
        return out


# -----------------------------
# Embeddings (modality + time)
# -----------------------------

class TimePositionalEmbedding(nn.Module):
    """Learned 1D positional embedding up to max_len.
    If T > max_len, it slices cyclically.
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(max_len, d_model)
        _init_weights(self.emb)

    def forward(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(T, device=device) % self.max_len
        pe = self.emb(idx)  # [T, D]
        return pe.unsqueeze(0).expand(B, T, -1)  # [B, T, D]


class ModalityEmbedding(nn.Module):
    def __init__(self, num_modalities: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(num_modalities, d_model)
        _init_weights(self.emb)

    def forward(self, m_index: int, B: int, T: int, device: torch.device) -> torch.Tensor:
        e = self.emb(torch.tensor([m_index], device=device))  # [1, D]
        return e.unsqueeze(1).expand(B, T, -1)  # [B, T, D]
import torch
import torch.nn as nn
import math

class Learned2DPos(nn.Module):

    def __init__(self, max_H: int, max_W: int, d_model: int, std: float = 0.02):
        super().__init__()
        self.max_H, self.max_W = max_H, max_W
        self.row = nn.Embedding(max_H, d_model)
        self.col = nn.Embedding(max_W, d_model)
        nn.init.normal_(self.row.weight, std=std)
        nn.init.normal_(self.col.weight, std=std)

    def forward(self, B: int, H: int, W: int, device=None, flatten: bool = True):
        """
        Returns:
          [B, H*W, D] if flatten=True (row-major), else [B, H, W, D]
        """
        device = device if device is not None else self.row.weight.device
        r = torch.arange(H, device=device) % self.max_H          # [H]
        c = torch.arange(W, device=device) % self.max_W          # [W]
        pos = self.row(r).unsqueeze(1) + self.col(c).unsqueeze(0)  # [H, W, D]
        if flatten:
            pos = pos.reshape(1, H * W, -1).expand(B, -1, -1)      # [B, HW, D]
        else:
            pos = pos.unsqueeze(0).expand(B, -1, -1, -1)           # [B, H, W, D]
        return pos


# -----------------------------
# Attention blocks
# -----------------------------

class MHA(nn.Module):
    def __init__(self, d: int, n_heads: int, p_drop: float = 0.1, cross: bool = False):
        super().__init__()
        self.cross = cross
        self.q_ln = nn.LayerNorm(d)
        self.kv_ln = nn.LayerNorm(d) if cross else None
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=p_drop, batch_first=True)
        self.drop = nn.Dropout(p_drop)

    def forward(
        self,
        x_q: torch.Tensor,                       # [B, L_q, D]
        x_kv: Optional[torch.Tensor] = None,     # [B, N, D] (cross일 때만)
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, N], True=mask
        attn_mask: Optional[torch.Tensor] = None           # [L_q, N] or [B, L_q, N]
    ) -> torch.Tensor:
        if self.cross:
            assert x_kv is not None, "cross-attn needs x_kv"
            q = self.q_ln(x_q)
            kv = self.kv_ln(x_kv)
            k = v = kv
        else:
            q = self.q_ln(x_q)
            k = v = q

        out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        return x_q + self.drop(out)  # pre-norm + 잔차



class FFN(nn.Module):
    def __init__(self, d: int, expansion: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * expansion)
        self.fc2 = nn.Linear(d * expansion, d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return x + self.drop(h)


# -----------------------------
# Fuse-MoE (simplified / practical)
# -----------------------------

class TopKRouter(nn.Module):
    """Simple top-k router with optional noise (Switch/MoE style).
    Returns gates and expert indices for each token.
    """
    def __init__(self, d: int, n_experts: int, k: int = 2, noisy: bool = True):
        super().__init__()
        self.noisy = noisy
        self.k = k
        self.w = nn.Linear(d, n_experts)
        _init_weights(self.w)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        logits = self.w(x)  # [B, T, E]
        if self.noisy and self.training:
            logits = logits + torch.randn_like(logits) * 0.5
        topk_vals, topk_idx = torch.topk(logits, k=self.k, dim=-1)  # [B, T, k]
        gates = F.softmax(topk_vals, dim=-1)  # [B, T, k]
        return gates, topk_idx


class ExpertFFN(nn.Module):
    def __init__(self, d: int, expansion: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, d * expansion)
        self.fc2 = nn.Linear(d * expansion, d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class FuseMoE(nn.Module):
    """Fuse-MoE-like layer: per-token routing to top-k experts with weighted combine.

    In the original Fuse-MoE (medical MM), routing considers modality/quality. Here, inputs
    already carry modality/time embeddings; router therefore learns to exploit them.
    """
    def __init__(self, d: int, n_experts: int = 8, k: int = 2, expansion: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.router = TopKRouter(d, n_experts, k=k, noisy=True)
        self.experts = nn.ModuleList([ExpertFFN(d, expansion, p_drop) for _ in range(n_experts)])
        self.out_drop = nn.Dropout(p_drop)
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN then route each token independently
        h = self.ln(x)
        gates, idx = self.router(h)        # gates: [B,T,k], idx: [B,T,k]
        B, T, k = gates.shape
        D = h.size(-1)

        # Gather expert outputs
        # Compute for each unique expert to avoid redundant compute
        h_flat = h.reshape(B * T, D)
        idx_flat = idx.reshape(B * T, k)
        gates_flat = gates.reshape(B * T, k)

        # For simplicity, compute all experts then gather (efficient for moderate E)
        expert_outs = [self.experts[e](h_flat) for e in range(len(self.experts))]  # list of [BT, D]
        expert_stack = torch.stack(expert_outs, dim=1)  # [BT, E, D]

        # Gather top-k per token
        gathered = torch.gather(expert_stack, 1, idx_flat.unsqueeze(-1).expand(-1, -1, D))  # [BT,k,D]
        mixed = (gathered * gates_flat.unsqueeze(-1)).sum(dim=1)  # [BT, D]
        mixed = mixed.view(B, T, D)
        return x + self.out_drop(mixed)


# -----------------------------
# Perceiver-style latent fusion block with MoE FFN
# -----------------------------

class LatentBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, p_drop: float = 0.1, use_moe: bool = True, n_experts: int = 8):
        super().__init__()
        self.self_attn = MHA(d, n_heads, p_drop=p_drop, cross=False)
        self.cross_attn = MHA(d, n_heads, p_drop=p_drop, cross=True)
        self.ff = FuseMoE(d, n_experts=n_experts, k=2, expansion=4, p_drop=p_drop) if use_moe else FFN(d, expansion=4, p_drop=p_drop)

    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, N] (True=mask)
        attn_mask: Optional[torch.Tensor] = None,         # [B or 1, L_q, N] (optional)
    ) -> torch.Tensor:
        latents = self.self_attn(latents)  # latent self-attn (마스크 불필요)
        latents = self.cross_attn(latents, tokens, key_padding_mask=key_padding_mask, attn_mask=attn_mask)  # ← 마스크 전달
        latents = self.ff(latents)
        return latents



# -----------------------------
# Main Fleximodal Fusion Model
# -----------------------------

@dataclass
class FlexiConfig:
    d_model: int =256
    n_heads: int = 8
    n_layers: int = 6
    n_experts: int = 8
    latent_len: int = 256
    p_drop: float = 0.1
    num_classes: int = 2


class FleximodalFuseMoE(nn.Module):

    def __init__(self, modalities: List[str], proj_cfgs: Dict[str, ProjectionCfg], cfg: FlexiConfig):
        super().__init__()
        self.modalities = modalities
        self.mod2idx = {m: i for i, m in enumerate(modalities)}
        self.cfg = cfg

        self.proj_bank = ProjectionBank(cfg.d_model, modalities, proj_cfgs, p_drop=cfg.p_drop)

        max_len = max(proj_cfgs[m].max_len for m in modalities)
        self.time_pos = nn.ModuleDict({m: TimePositionalEmbedding(proj_cfgs[m].max_len, cfg.d_model) for m in modalities})
        self.image_pos2d = Learned2DPos(max_H=32, max_W=32, d_model=cfg.d_model) 
        self.mod_emb = ModalityEmbedding(num_modalities=len(modalities), d_model=cfg.d_model)

        # Latents with a dedicated CLS slot at index 0
        self.latents = nn.Parameter(torch.randn(cfg.latent_len, cfg.d_model) / math.sqrt(cfg.d_model))

        self.blocks = nn.ModuleList([
            LatentBlock(cfg.d_model, cfg.n_heads, p_drop=cfg.p_drop, use_moe=True, n_experts=cfg.n_experts)
            for _ in range(cfg.n_layers)
        ])
        self.norm_out = RMSNorm(cfg.d_model)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Dropout(cfg.p_drop),
            nn.Linear(cfg.d_model, cfg.num_classes)
        )
        self.apply(_init_weights)

    def _build_tokens(self, feats: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]):
        """
        Returns:
        tokens: FloatTensor [B, N, D]
        key_padding_mask: BoolTensor [B, N]  (True=패딩/결측)
        """
        tokens_list, mask_list = [], []
        device = next(self.parameters()).device

 
        for m, x in self.proj_bank(feats).items():
            B, T, D = x.shape

            x = x + self.time_pos[m](B, T, device)
            x = x + self.mod_emb(self.mod2idx[m], B, T, device)
            tokens_list.append(x)



            km = ~masks[m]                
            mask_list.append(km)

        if len(tokens_list) == 1:
            tokens = tokens_list[0]
            key_padding_mask = mask_list[0]
        else:
            tokens = torch.cat(tokens_list, dim=1)               # [B, N, D]
            key_padding_mask = torch.cat(mask_list, dim=1)       # [B, N]

        return tokens, key_padding_mask


    def forward(self, feats: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = next(iter(feats.values())).size(0)
        device = next(self.parameters()).device

        tokens, key_padding_mask = self._build_tokens(feats, masks)          # [B,N,D], [B,N]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1).to(device)     # [B,L,D]

        for blk in self.blocks:
            # Latents attend to tokens with mask
            latents = blk(latents, tokens, key_padding_mask=key_padding_mask)

        latents = self.norm_out(latents)
        cls = latents[:, 0, :]
        logits = self.head(cls)
        return {"logits": logits, "cls": cls}



# -----------------------------
# Minimal usage example (pseudo)
# -----------------------------
if __name__ == "__main__":
    # Define modalities and their incoming feature dims
    modalities = ["image", "audio", "timeseries", "text", "video"]
    proj_cfgs = {
        "image": ProjectionCfg(in_dim=1024, max_len=64),    # e.g., 8x8 ViT patch tokens
        "audio": ProjectionCfg(in_dim=768, max_len=400),     # e.g., BEATs frame tokens
        "timeseries": ProjectionCfg(in_dim=256, max_len=256),
        "text": ProjectionCfg(in_dim=768, max_len=128),
        "video": ProjectionCfg(in_dim=1024, max_len=128),
    }

    cfg = FlexiConfig(d_model=512, n_heads=8, n_layers=4, n_experts=8, latent_len=128, num_classes=2)
    model = FleximodalFuseMoE(modalities, proj_cfgs, cfg)

    B = 2
    feats = {
        "image": torch.randn(B, 64, 1024),
        "audio": torch.randn(B, 300, 768),  # shorter than max_len is ok
        # Missing some modalities is fine; just omit their keys here.
    }
    
    out = model(feats)
    print(out["logits"], out["cls"])  # [B, C], [B, D]
