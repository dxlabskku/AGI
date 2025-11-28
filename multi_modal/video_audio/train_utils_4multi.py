import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('/data/jupyter/AGI/fusion')

from multi_fusion import FleximodalFuseMoE, FlexiConfig, ProjectionCfg
from typing import Dict
import json 


CODE2IDX = {"A":0, "B1":1, "B2":2, "B4":3, "B5":4, "B6":5, "G":6}
NUM_CLASSES = 2


import os
import torch
import numpy as np

import os, json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List

# =========================
# 1) 로더: 두 포맷 자동 처리
# =========================
import os, json
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

# =============== 공통 유틸 ===============
def _to_2d_float(x) -> torch.Tensor:

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    x = x.float()

    if x.ndim == 1:
        x = x.unsqueeze(0)             
    elif x.ndim >= 3:
        T = x.shape[0]
        x = x.reshape(T, -1)         

    return x.contiguous()


def _safe_scalar_long(y_like) -> torch.Tensor:
  
    try:
        if isinstance(y_like, torch.Tensor):
            return y_like.view(-1)[0].to(torch.long)
        elif isinstance(y_like, (np.ndarray, list, tuple)):
            arr = np.array(y_like).reshape(-1)
            return torch.tensor(int(arr[0]) if arr.size > 0 else -1, dtype=torch.long)
        else:
            return torch.tensor(int(y_like), dtype=torch.long)
    except Exception:
        return torch.tensor(-1, dtype=torch.long)

def _to_multilabel_float(y_like) -> torch.Tensor:
    
    if isinstance(y_like, torch.Tensor):
        y = y_like.float().view(-1)
    else:
        y = torch.tensor(np.array(y_like, dtype=np.float32).reshape(-1), dtype=torch.float32)
    return y

def pad_time(x: torch.Tensor, T_max: int):
    # x: [T, D]
    if x.size(0) == T_max:
        return x
    out = x.new_zeros((T_max, x.size(1)))
    out[:x.size(0)] = x
    return out

# =========================
# 1) 로더: 세 포맷 자동 처리
# =========================
def load_pt_as_feats_and_label(path: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, str]:

    ckpt = torch.load(path, map_location="cpu")


    if ("img_feat" in ckpt or "image_feat" in ckpt) and ("text_feat" in ckpt or "txt_feat" in ckpt):
        img_raw = ckpt.get("img_feat", ckpt.get("image_feat"))
        txt_raw = ckpt.get("text_feat", ckpt.get("txt_feat"))
        img = _to_2d_float(img_raw)
        txt = _to_2d_float(txt_raw)

  
        raw_label = ckpt.get("label_bin", ckpt.get("labels", None))
        if raw_label is None:  # 라벨 누락이면 빈 벡터 대신 -1 싱글라벨로 표기
            y = torch.tensor(-1, dtype=torch.long)
        else:
        
            y = _to_multilabel_float(raw_label)   # [C] float

        uid = ckpt.get("patient_id", ckpt.get("filename", os.path.basename(path)))
        feats = {"image": img, "text": txt}
        return feats, y, uid

 
    if "audio_tokens" in ckpt or "video_tokens" in ckpt:
        a = ckpt.get("audio_tokens", None)
        v = ckpt.get("video_tokens", None)
        if a is None or v is None:
            raise ValueError(f"Missing audio_tokens or video_tokens in: {path}")

        # audio
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        if a.ndim == 3 and a.shape[1] == 1:
            a = a.squeeze(1)  # (T,1,512)->(T,512)
        a = _to_2d_float(a)

        # video
        v = _to_2d_float(v)

        labels = ckpt.get("labels", {})
        if isinstance(labels, dict) and "binary" in labels:
            y = _safe_scalar_long(labels["binary"])
        else:
            y = _safe_scalar_long(labels)

        uid = ckpt.get("id", os.path.basename(path))
        feats = {"audio": a, "video": v}
        return feats, y, uid


    if "embedding" in ckpt:
        emb = _to_2d_float(ckpt["embedding"])

        raw_label = ckpt.get("label", None)
        if raw_label is None:
            y = torch.tensor(-1, dtype=torch.long)
        else:
        
            y = _safe_scalar_long(raw_label)

        uid = ckpt.get("filename", os.path.basename(path))
        feats = {"series": emb}
        return feats, y, uid

    raise ValueError(f"Unsupported .pt format: {path}")


# =========================
# 2) Dataset / Collate
# =========================
class PTListDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self.paths = json.load(f)
        assert isinstance(self.paths, list) and len(self.paths) > 0 and all(isinstance(p, str) for p in self.paths), \
            "JSON must be a non-empty list of path strings."

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        feats, y, uid = load_pt_as_feats_and_label(self.paths[i])
        for k in feats:
            feats[k] = feats[k].float().contiguous()
 
        return feats, y, uid


def collate_mm(batch):
  

    all_keys = sorted({k for feats, _, _ in batch for k in feats.keys()})

    feats_out, masks = {}, {}


    mod_dims = {}
    for mk in all_keys:
        for feats, _, _ in batch:
            if mk in feats:
                mod_dims[mk] = int(feats[mk].size(-1))
                break


    for mk in all_keys:
        seqs = []
        lens = []
        D = mod_dims[mk]

        for feats, _, _ in batch:
            if mk in feats:
                x = feats[mk]                  # [T, D]
                lens.append(x.size(0))
                seqs.append(x)
            else:
      
                x = torch.zeros(0, D, dtype=torch.float32)
                lens.append(0)
                seqs.append(x)

        T_max = max(lens) if len(lens) > 0 else 0
        if T_max == 0:
  
            stacked = torch.zeros(len(batch), 1, D, dtype=torch.float32)
            mask = torch.zeros(len(batch), 1, dtype=torch.bool)
        else:
            padded = []
            for x in seqs:
                T = x.size(0)
                if T < T_max:
                    pad = torch.zeros(T_max - T, D, dtype=x.dtype)
                    x = torch.cat([x, pad], dim=0)
                padded.append(x)
            stacked = torch.stack(padded, dim=0)  # [B, T_max, D]

            mask = torch.zeros(len(batch), T_max, dtype=torch.bool)
            for i, L in enumerate(lens):
                if L > 0:
                    mask[i, :L] = True

        feats_out[mk] = stacked.contiguous()
        masks[mk] = mask


    ys = [b[1] for b in batch]

    def _is_multilabel_tensor(y):
        return isinstance(y, torch.Tensor) and y.dtype in (torch.float32, torch.float64) and y.ndim >= 1 and y.numel() > 1

    has_multi = any(_is_multilabel_tensor(y) for y in ys)

    if has_multi:
     
        C = next(int(y.numel()) for y in ys if _is_multilabel_tensor(y))
        Yf = []
        for y in ys:
            if _is_multilabel_tensor(y):
                vec = y.flatten().float()
                if vec.numel() != C:
                    raise ValueError(f"Inconsistent multilabel size: {vec.numel()} vs {C}")
                Yf.append(vec)
            else:
           
                cls = int(y.item()) if isinstance(y, torch.Tensor) else int(y)
                vec = torch.zeros(C, dtype=torch.float32)
                if 0 <= cls < C:
                    vec[cls] = 1.0
                Yf.append(vec)
        Y = torch.stack(Yf, dim=0)  # [B, C] float
    else:

        Y = torch.stack([
            (y.view(()) if isinstance(y, torch.Tensor) else torch.tensor(int(y)).view(()))
            for y in ys
        ], dim=0).long()            # [B]

    uids = [b[2] for b in batch]
    return feats_out, masks, Y, uids




def infer_modal_dims_from_dataset(json_path: str, probe_n: int = 128):
    import random, json, os
    with open(json_path, "r") as f:
        paths = json.load(f)


    dims = {}  # mk -> D
    for p in paths:
        feats, _, _ = load_pt_as_feats_and_label(p)
        for mk, x in feats.items():
            D = int(x.size(-1))
            if mk not in dims:
                dims[mk] = D
            elif dims[mk] != D:
                raise ValueError(f"Inconsistent dim for {mk}: seen {dims[mk]} vs {D} ({p})")
    print('dims',dims)
    return dims

def build_model_from_dataset(json_path: str, num_classes: int):
    dims = infer_modal_dims_from_dataset(json_path)
    modalities = sorted(dims.keys())
    proj_cfgs = {mk: ProjectionCfg(in_dim=D, max_len=4000) for mk, D in dims.items()}
    cfg = FlexiConfig(
        d_model=256, n_heads=8, n_layers=4, n_experts=8,
        latent_len=128, p_drop=0.1, num_classes=num_classes
    )
    return FleximodalFuseMoE(modalities, proj_cfgs, cfg)




import torch
import numpy as np

@torch.no_grad()
def binary_metrics_from_logits(logits: torch.Tensor,
                               targets: torch.Tensor,
                               thr: float = 0.5,
   
 
    if logits.ndim == 2 and logits.size(1) == 2:
      
        probs = torch.softmax(logits, dim=1)[:, 1]
    elif logits.ndim == 2 and logits.size(1) == 1:
        probs = torch.sigmoid(logits.squeeze(1))
    else:
   
        probs = torch.sigmoid(logits)


    y = targets.view(-1).float()


    preds = (probs >= thr).float()

  
    tp = (preds * y).sum().item()
    tn = ((1 - preds) * (1 - y)).sum().item()
    fp = (preds * (1 - y)).sum().item()
    fn = ((1 - preds) * y).sum().item()
    total = tp + tn + fp + fn


    acc  = (tp + tn) / max(1.0, total)
    prec = tp / max(eps, (tp + fp))
    rec  = tp / max(eps, (tp + fn))
    f1   = 2 * prec * rec / max(eps, (prec + rec))

    return {
        "acc":   float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "thr": float(thr)
    }





