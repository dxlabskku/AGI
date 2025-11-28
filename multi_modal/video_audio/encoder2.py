import os

# 안전한 tmp/cache 경로 지정
os.environ["TMPDIR"] = "/data/tmp"
os.environ["HF_HOME"] = "/data/cache/hf"
os.environ["TRANSFORMERS_CACHE"] = "/data/cache/hf"
os.environ["TORCH_HOME"] = "/data/cache/torch"

# 폴더가 없으면 생성
for p in ["/data/tmp", "/data/cache/hf", "/data/cache/torch"]:
    os.makedirs(p, exist_ok=True)


from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

from transformers import ClapProcessor, ClapModel

# -----------------------------
# Video: Swin3D (torchvision)
# -----------------------------
@dataclass
class VideoSwinCfg:
    model_name: str = "swin3d_t"
    pretrained: bool = True
    frames: int = 16
    img_size: int = 224
    stride: int = 4
    return_seq: bool = False         # True면 per-temporal token 반환
    pool_type: str = "avg"           # (seq 모드서만 사용)

class VideoSwinBackbone(nn.Module):
    """
    Input:  frames [B, T, 3, H, W], 값 범위 [0,1] 또는 [0,255]
    Output:
      - return_seq=False: [B, D]
      - return_seq=True : [B, T', D] (stage4 출력을 h,w 평균 후)
    """
    def __init__(self, cfg: VideoSwinCfg = VideoSwinCfg()):
        super().__init__()
        self.cfg = cfg
        weights = Swin3D_T_Weights.KINETICS400_V1 if cfg.pretrained else None

        self.model = swin3d_t(weights=weights)
        # 분류헤드 제거 → forward가 [B, D] 리턴
        self.model.head = nn.Identity()

        # out_dim 안전 추출
        if hasattr(self.model, "norm") and hasattr(self.model.norm, "normalized_shape"):
            self.out_dim = int(self.model.norm.normalized_shape[0])
        else:
            # 아주 드문 케이스 fallback
            self.out_dim = 768

        # return_seq용: stage 출력 훅
        self._feat_5 = None
        def _hook(m, x, y):
            # y: [B, C, t, h, w]
            self._feat_5 = y
        # 마지막 stage 블록의 출력 register (features[-1]는 Sequential)
        self.model.features[-1].register_forward_hook(_hook)

        # mean/std (weights.meta에서 가져옴)
        if weights is not None and hasattr(weights, "meta"):
            meta = weights.meta
            self.mean = torch.tensor(meta.get("mean", [0.45, 0.45, 0.45])).view(1, 1, 3, 1, 1)
            self.std  = torch.tensor(meta.get("std",  [0.225,0.225,0.225])).view(1, 1, 3, 1, 1)
        else:
            self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1,1,3,1,1)
            self.std  = torch.tensor([0.225,0.225,0.225]).view(1,1,3,1,1)

    def _normalize(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames
        if x.dtype != torch.float32:
            x = x.float()
        # [0,255] → [0,1]
        if x.max() > 1.5:
            x = x / 255.0
        mean = self.mean.to(x.device, x.dtype)
        std  = self.std.to(x.device, x.dtype)
        return (x - mean) / std

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: [B, T, 3, H, W]
        """
        assert frames.dim() == 5 and frames.size(2) == 3, "frames must be [B,T,3,H,W]"
        # Swin3D expects [B, C, T, H, W]
        x = frames.permute(0, 2, 1, 3, 4).contiguous()
        x = self._normalize(x.permute(0,2,1,3,4)).permute(0,2,1,3,4)  # 정규화는 [B,T,3,H,W] 기준으로 적용 후 복원

        # forward: head=Identity → [B, D]
        with torch.no_grad():
            pooled = self.model(x)  # [B, D]

        if not self.cfg.return_seq:
            return pooled

        # 훅으로 받은 stage5 출력을 시간토큰으로 변환
        feats = self._feat_5  # [B, C, t, h, w]
        assert feats is not None, "Feature hook not captured."
        if self.cfg.pool_type == "avg":
            t_tokens = feats.mean(dim=[3,4]).transpose(1, 2)  # [B, t, C]
        else:
            t_tokens = feats.amax(dim=[3,4]).transpose(1, 2)  # [B, t, C]
        return t_tokens  # [B, T', D==C]

# -----------------------------
# Audio: CLAP (HTSAT-fused) windowed embeddings
# -----------------------------
@dataclass
class CLAPAudioCfg:
    model_name: str = "laion/clap-htsat-fused"
    sr: int = 48000
    win_sec: float = 1.0
    hop_sec: float = 0.5
    normalize: bool = True

class AudioCLAPWindowed(nn.Module):
    """
    Input: wav [B, N] @ sr
    Output: [B, T, D]  (각 윈도우 토큰)
    """
    def __init__(self, cfg: CLAPAudioCfg = CLAPAudioCfg()):
        super().__init__()
        self.cfg = cfg
        self.processor = ClapProcessor.from_pretrained(cfg.model_name)
        self.model = ClapModel.from_pretrained(cfg.model_name)
        # proj dim (없으면 hidden_size)
        self.out_dim = getattr(self.model.config, "projection_dim",
                               getattr(self.model.config, "hidden_size"))

    def _frame_audio_list(self, wav: torch.Tensor):
        """
        Returns:
          flat_list: List[np.ndarray] 길이 sum_T (B의 모든 윈도우를 1차로)
          T_list:    각 배치별 윈도우 개수 리스트 [T_b1, T_b2, ...]
        """
        B, N = wav.shape
        W = int(self.cfg.win_sec * self.cfg.sr)
        H = int(self.cfg.hop_sec * self.cfg.sr)
        flat_list = []
        T_list = []
        for b in range(B):
            x = wav[b]
            segs = []
            for s in range(0, max(1, N - W + 1), H):
                if s + W <= N:
                    seg = x[s:s+W]
                else:
                    pad = s + W - N
                    seg = F.pad(x[s:], (0, pad))
                segs.append(seg)
            T_list.append(len(segs))
            flat_list.extend([seg.cpu().numpy() for seg in segs])
        return flat_list, T_list

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [B, N] (float, -1~1 권장)
        """
        device = wav.device
        self.model.to(device)
        self.model.eval()

        flat_audios, T_list = self._frame_audio_list(wav)
        with torch.no_grad():
            inputs = self.processor(audios=flat_audios,
                                    sampling_rate=self.cfg.sr,
                                    return_tensors="pt",
                                    padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = self.model.get_audio_features(**inputs)  # [sum_T, D]

        # [sum_T, D] → [B, T, D]
        splits = torch.split(out, T_list, dim=0)
        feats = torch.stack(splits, dim=0)  # [B, T, D]
        if self.cfg.normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

torch.set_grad_enabled(False)

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # Video
    vcfg = VideoSwinCfg(pretrained=True, return_seq=False)
    vbackbone = VideoSwinBackbone(vcfg).cuda()
    frames = torch.rand(2, 16, 3, 224, 224, device="cuda")  # [0,1]
    vt = vbackbone(frames)  # [B, D] or [B, T', D] if return_seq=True
    print("Video tokens:", vt.shape)

    # Audio
    aud = AudioCLAPWindowed(CLAPAudioCfg()).cuda()
    wav = torch.randn(2, 48000 * 5, device="cuda")
    at = aud(wav)  # [B, T, D]
    print("Audio tokens:", at.shape)
