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
    """
    임의 입력(x: Tensor/ndarray/리스트)을 [T, D] float 텐서로 강제 변환.
    - 1D: [D] → [1, D]
    - 2D: 그대로 유지
    - 3D 이상: 첫 축을 T로 두고 나머지 평탄화 [T, D]
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    x = x.float()

    if x.ndim == 1:
        x = x.unsqueeze(0)             # [D] → [1, D]  ✅ 수정 포인트
    elif x.ndim >= 3:
        T = x.shape[0]
        x = x.reshape(T, -1)           # [T, ...] → [T, D]
    # 이제 2D 보장
    return x.contiguous()


def _safe_scalar_long(y_like) -> torch.Tensor:
    """
    싱글라벨을 Long scalar로 변환. 변환 실패 시 -1.
    """
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
    """
    멀티라벨을 [C] float 텐서로 변환 (0/1, 또는 확률).
    """
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
    """
    지원 포맷:
    A) 비디오-오디오 페어:
        {
          "audio_tokens": Tensor (T,1,512) or (T,512),
          "video_tokens": Tensor (T,7,8) 등,
          "labels": {"binary": 0 or 1} 또는 정수,
          "id": "..." (optional)
        }
    B) 시계열 단독 (series):
        {
          "embeddings": Tensor/ndarray [T, D],
          "labels": Tensor/ndarray/scalar or None,
          "filename": "..." (optional),
        }
    C) 이미지-텍스트 멀티라벨:
        {
          "img_feat": Tensor/ndarray [T_img, D_img] 또는 [D_img] (자동 2D화),
          "text_feat": Tensor/ndarray [T_txt, D_txt] 또는 [D_txt],
          "label": Iterable/ndarray/Tensor 길이 C의 멀티핫(0/1) 또는 확률,
          "id" 또는 "filename": "..." (optional)
        }

    Returns:
        feats: {"audio":..., "video":...} or {"series":...} or {"image":..., "text":...}
        y:    LongTensor scalar (싱글라벨) 또는 FloatTensor [C] (멀티라벨)
        uid:  str
    """
    ckpt = torch.load(path, map_location="cpu")

    # ---- Case C: 이미지-텍스트 멀티라벨 ----
    if ("img_feat" in ckpt or "image_feat" in ckpt) and ("text_feat" in ckpt or "txt_feat" in ckpt):
        img_raw = ckpt.get("img_feat", ckpt.get("image_feat"))
        txt_raw = ckpt.get("text_feat", ckpt.get("txt_feat"))
        img = _to_2d_float(img_raw)
        txt = _to_2d_float(txt_raw)

        # 멀티라벨(label 또는 labels 키 모두 지원)
        raw_label = ckpt.get("label_bin", ckpt.get("labels", None))
        if raw_label is None:  # 라벨 누락이면 빈 벡터 대신 -1 싱글라벨로 표기
            y = torch.tensor(-1, dtype=torch.long)
        else:
            # 멀티라벨로 간주
            y = _to_multilabel_float(raw_label)   # [C] float

        uid = ckpt.get("patient_id", ckpt.get("filename", os.path.basename(path)))
        feats = {"image": img, "text": txt}
        return feats, y, uid

    # ---- Case A: 오디오-비디오 페어 ----
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

    # ---- Case B: 시계열 단독 ----
    if "embedding" in ckpt:
        emb = _to_2d_float(ckpt["embedding"])

        raw_label = ckpt.get("label", None)
        if raw_label is None:
            y = torch.tensor(-1, dtype=torch.long)
        else:
            # 시계열 단독은 기본적으로 싱글라벨로 해석
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
        # y는 싱글라벨(Long scalar) 또는 멀티라벨(Float [C]) 그대로 유지
        return feats, y, uid


def collate_mm(batch):
    """
    batch: List[(feats_dict, y, uid)]
    returns:
      feats_out: Dict[str, FloatTensor[B, T_max, D]]
      masks:     Dict[str, BoolTensor[B, T_max]]  # 유효길이 마스크
      Y:         Long[B] (싱글) 또는 Float[B, C] (멀티)
      uids:      List[str]
    """
    # --- 배치 내 모든 모달의 합집합 ---
    all_keys = sorted({k for feats, _, _ in batch for k in feats.keys()})

    feats_out, masks = {}, {}

    # --- 모달별 D 추론 (해당 모달이 있는 첫 샘플에서) ---
    mod_dims = {}
    for mk in all_keys:
        for feats, _, _ in batch:
            if mk in feats:
                mod_dims[mk] = int(feats[mk].size(-1))
                break

    # --- 모달별 패딩/스택 ---
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
                # 해당 샘플에 모달이 없으면 길이 0로 취급
                x = torch.zeros(0, D, dtype=torch.float32)
                lens.append(0)
                seqs.append(x)

        T_max = max(lens) if len(lens) > 0 else 0
        if T_max == 0:
            # 전부 길이 0이면 [B, 1, D] 제로로 두고 마스크는 False
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

    # --- 라벨 스택 ---
    ys = [b[1] for b in batch]

    def _is_multilabel_tensor(y):
        return isinstance(y, torch.Tensor) and y.dtype in (torch.float32, torch.float64) and y.ndim >= 1 and y.numel() > 1

    has_multi = any(_is_multilabel_tensor(y) for y in ys)

    if has_multi:
        # C 추론
        C = next(int(y.numel()) for y in ys if _is_multilabel_tensor(y))
        Yf = []
        for y in ys:
            if _is_multilabel_tensor(y):
                vec = y.flatten().float()
                if vec.numel() != C:
                    raise ValueError(f"Inconsistent multilabel size: {vec.numel()} vs {C}")
                Yf.append(vec)
            else:
                # 싱글라벨을 one-hot로 승격
                cls = int(y.item()) if isinstance(y, torch.Tensor) else int(y)
                vec = torch.zeros(C, dtype=torch.float32)
                if 0 <= cls < C:
                    vec[cls] = 1.0
                Yf.append(vec)
        Y = torch.stack(Yf, dim=0)  # [B, C] float
    else:
        # 전부 싱글라벨
        Y = torch.stack([
            (y.view(()) if isinstance(y, torch.Tensor) else torch.tensor(int(y)).view(()))
            for y in ys
        ], dim=0).long()            # [B]

    uids = [b[2] for b in batch]
    return feats_out, masks, Y, uids


# =========================
# 3) 모델 생성
# =========================
# 가정: ProjectionCfg, FlexiConfig, FleximodalFuseMoE 는 기존과 동일하게 사용

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
                               eps: float = 1e-8):
    """
    logits: [B], [B,1], 또는 [B,2]
        - [B,2]: 2-클래스 로짓 (CELoss) -> softmax로 양성(=1) 확률 사용
        - [B], [B,1]: 단일 로짓 (BCEWithLogitsLoss) -> sigmoid 사용
    targets: [B] in {0,1} (long/float 모두 허용)
    Returns: dict of binary classification metrics
    """
    # 확률 산출
    if logits.ndim == 2 and logits.size(1) == 2:
        # 2-클래스 로짓 -> softmax 확률의 positive(=1) 클래스
        probs = torch.softmax(logits, dim=1)[:, 1]
    elif logits.ndim == 2 and logits.size(1) == 1:
        probs = torch.sigmoid(logits.squeeze(1))
    else:
        # [B] 단일 로짓
        probs = torch.sigmoid(logits)

    # 타깃 정리
    y = targets.view(-1).float()

    # 예측 이진화
    preds = (probs >= thr).float()

    # 혼동행렬 성분
    tp = (preds * y).sum().item()
    tn = ((1 - preds) * (1 - y)).sum().item()
    fp = (preds * (1 - y)).sum().item()
    fn = ((1 - preds) * y).sum().item()
    total = tp + tn + fp + fn

    # 지표
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


'''
import os
import json

def save_pt_paths_to_json(root_dir, output_json):
    pt_paths = []

    # 모든 하위 디렉토리 탐색
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".pt"):
                full_path = os.path.join(dirpath, filename)
                pt_paths.append(full_path)

    # JSON으로 저장
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pt_paths, f, indent=4, ensure_ascii=False)

    print(f"✅ {len(pt_paths)}개의 .pt 파일 경로가 {output_json}에 저장되었습니다.")


# 예시 실행
root_dir = "/data/jupyter/AGI/datasets/XD-violence/video_feat/unsplit"  # 탐색할 폴더 경로
output_json = "/data/jupyter/AGI/datasets/XD-violence/video_feat/json/video_json.json"
save_pt_paths_to_json(root_dir, output_json)
'''


