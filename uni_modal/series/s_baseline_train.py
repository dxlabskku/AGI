#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Series TS2Vec Baseline for SKAB (uni-modal time series) → d=256 embeddings → anomaly score
- Self-supervised TS2Vec-style training with temporal augmentations
- Projection head to d=256
- Normal memory bank from train → kNN/KDE anomaly scoring
- Threshold via KDE crossing on validation

Usage examples
--------------
# 1) Train self-supervised encoder on normal-only train windows
python series_ts2vec_baseline.py train \
  --data_root /path/to/SKAB \
  --window 256 --stride 64 --batch 128 --epochs 80

# 2) Build memory bank from train embeddings (saves knn + kde)
python series_ts2vec_baseline.py build_bank \
  --data_root /path/to/SKAB --window 256 --stride 64

# 3) Evaluate on test split (outputs CSV + metrics)
python series_ts2vec_baseline.py eval \
  --data_root /path/to/SKAB --window 256 --stride 64 \
  --score_head kde --smooth 5 --out_dir ./outputs

Dataset expectation (SKAB)
--------------------------
- data_root/
    train/*.csv    # normal-only files preferred; if labels exist, they are ignored for training
    test/*.csv     # may contain anomalies with column name 'anomaly' or 'label' (0/1)
- CSV columns: time[, ...features..., label?]
- Label column (optional for train): one of ['anomaly','label','y'] (0/1)

Notes
-----
- If no label column is found in test, evaluation will run without metrics (scores only).
- This script is intentionally dependency-light: numpy, pandas, torch, sklearn.
"""
from __future__ import annotations
import os, math, argparse, json, random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    out = (cumsum[k:] - cumsum[:-k]) / float(k)
    pad = np.concatenate([np.repeat(out[0], k-1), out])
    return pad[: len(x)]


# ---------------------------------------------------------
# Data handling: SKAB
# ---------------------------------------------------------

# 파일 상단 근처에 추가
LABEL_CANDS = ['anomaly','label','y','class','Class','is_anomaly']
TIME_CANDS  = ['timestamp','time','datetime','date','index']

def read_skab_csv(path):
    # 1) 구분자 자동 추정
    df = pd.read_csv(path, sep=None, engine='python')
    # 2) 한 컬럼으로 읽혔으면 세미콜론으로 재시도
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=';', engine='python')

    # ★ Unnamed 컬럼 제거
    drop_unnamed = [c for c in df.columns if str(c).startswith('Unnamed')]
    if drop_unnamed:
        df = df.drop(columns=drop_unnamed)
        
    # 3) 라벨/시간 컬럼 처리
    y = None
    for c in LABEL_CANDS:
        if c in df.columns:
            y = df[c].astype(int).values
            df = df.drop(columns=[c])
            break
    for c in TIME_CANDS:
        if c in df.columns:
            df = df.drop(columns=[c])
    # 4) 숫자만 유지
    df = df.select_dtypes(include=[np.number])
    return df, y


LABEL_CANDIDATES = ["anomaly", "label", "y"]

def load_csv_with_optional_label(path: Path) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    df = pd.read_csv(path)
    label = None
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            label = df[c].astype(int).values
            df = df.drop(columns=[c])
            break
    # drop obvious non-feature columns
    for c in ["timestamp", "time", "datetime", "date"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df, label


def list_csvs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.csv") if p.is_file()])

def align_features(X: np.ndarray, expected_in: int) -> np.ndarray:
    # X: [N, L, F]
    F = X.shape[-1]
    if F == expected_in:
        return X
    elif F > expected_in:
        return X[..., :expected_in]          # 초과 컬럼 잘라내기
    else:
        # 부족하면 0-padding
        pad = np.zeros((X.shape[0], X.shape[1], expected_in - F), dtype=X.dtype)
        return np.concatenate([X, pad], axis=-1)


class SlidingWindowSeries(Dataset):
    def __init__(self, 
                 arrays: List[np.ndarray],
                 labels: Optional[List[np.ndarray]],
                 window: int, stride: int, 
                 normalize: bool = True,
                 return_labels: bool = False,
                 pos_ratio_threshold: float = 0.5):
        """
        arrays: list of [T, F] arrays (concatenates across files via sliding windows)
        labels: list of [T] arrays (0/1) or None
        return_labels: if True, returns y_window for each window
        pos_ratio_threshold: window is anomalous if mean(label)>threshold
        """
        self.window = window
        self.stride = stride
        self.return_labels = return_labels and (labels is not None)
        Xs, Ys = [], []
        for i, arr in enumerate(arrays):
            L = arr.shape[0]
            y = labels[i] if labels is not None else None
            # normalization per file using train statistics should be done outside for train
            for s in range(0, max(L - window + 1, 0), stride):
                seg = arr[s: s + window]
                Xs.append(seg)
                if self.return_labels:
                    ywin = y[s: s + window]
                    yflag = 1 if (ywin.mean() > pos_ratio_threshold) else 0
                    Ys.append(yflag)
        self.X = np.stack(Xs, axis=0).astype(np.float32) if Xs else np.zeros((0, window, arrays[0].shape[1]), dtype=np.float32)
        self.Y = np.array(Ys, dtype=np.int64) if self.return_labels and Ys else None
        # global normalize if requested (for non-train)
        if normalize:
            mu = self.X.reshape(-1, self.X.shape[-1]).mean(axis=0, keepdims=True)
            sd = self.X.reshape(-1, self.X.shape[-1]).std(axis=0, keepdims=True) + 1e-8
            self.X = (self.X - mu) / sd

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # [L, F]
        if self.return_labels:
            return x, self.Y[idx]
        else:
            return x


# ---------------------------------------------------------
# Augmentations for TS2Vec-like training
# ---------------------------------------------------------
class TimeSeriesAug:
    def __init__(self, jitter_p=0.3, scaling_p=0.2, time_mask_p=0.2, crop_p=0.3):
        self.jitter_p = jitter_p
        self.scaling_p = scaling_p
        self.time_mask_p = time_mask_p
        self.crop_p = crop_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        # B, L, F = x.shape
        B, L, C = x.shape
        out = x.clone()
        device = x.device
        # jitter
        if random.random() < self.jitter_p:
            noise = torch.randn_like(out) * 0.01
            out = out + noise
        # scaling
        if random.random() < self.scaling_p:
            # scale = (0.9 + 0.2 * torch.rand(B, 1, F, device=device))
            scale = (0.9 + 0.2 * torch.rand(B, 1, C, device=device))
            out = out * scale
        # time mask
        if random.random() < self.time_mask_p:
            mask_len = max(1, int(L * 0.1))
            start = random.randint(0, max(L - mask_len, 0))
            out[:, start:start+mask_len, :] = 0.0
        # random crop to a sub-length and then resize back via pad (invariance)
        if random.random() < self.crop_p:
            keep = random.randint(int(L*0.6), L)
            start = random.randint(0, L - keep)
            cropped = out[:, start:start+keep, :]
            # pad back to L at random side
            pad_left = random.randint(0, L - keep)
            pad_right = L - keep - pad_left
            out = F.pad(cropped, (0,0, pad_left, pad_right))  # ← 여기의 F는 functional 모듈 그대로 유지     
        return out


# ---------------------------------------------------------
# TS2Vec-like encoder (lightweight TCN + projection head)
# ---------------------------------------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, d=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=d*(k-1)//2, dilation=d)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.res = (in_ch == out_ch)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # x: [B, C, L]
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        y = self.dropout(y)
        if self.res:
            y = y + x
        return y

class TS2VecEncoder(nn.Module):
    def __init__(self, in_feat: int, hid: int = 256, depth: int = 4):
        super().__init__()
        chs = [in_feat, hid, hid, hid, hid]
        blocks = []
        for i in range(depth):
            blocks.append(TCNBlock(chs[i], chs[i+1], k=5, d=2**i if i<3 else 1))
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [B, L, F]
        x = x.transpose(1, 2)  # [B, F, L]
        h = self.net(x)        # [B, C, L]
        g = self.pool(h).squeeze(-1)  # [B, C]
        return g

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        z = self.fc2(self.act(self.fc1(x)))
        z = F.normalize(z, dim=-1)
        return z

class TS2VecModel(nn.Module):
    def __init__(self, in_feat: int, out_dim: int = 256):
        super().__init__()
        self.encoder = TS2VecEncoder(in_feat)
        self.proj = ProjectionHead(256, out_dim)

    def forward(self, x):
        g = self.encoder(x)
        z = self.proj(g)
        return z  # [B, d]


# ---------------------------------------------------------
# Contrastive loss (NT-Xent)
# ---------------------------------------------------------

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B, D = z1.shape
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.mm(z, z.t())  # [2B, 2B]
    # mask self-sim
    mask = torch.eye(2*B, device=z.device).bool()
    sim = sim / temp
    sim.masked_fill_(mask, -1e9)
    # positives: i <-> i+B and i+B <-> i
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)
    loss = F.cross_entropy(sim, pos)
    return loss


# ---------------------------------------------------------
# Training loop (self-supervised on normal windows)
# ---------------------------------------------------------
@dataclass
class TrainConfig:
    data_root: Path
    window: int = 256
    stride: int = 64
    batch: int = 128
    epochs: int = 80
    lr: float = 1e-3
    wd: float = 1e-4
    temp: float = 0.1
    seed: int = 42
    out_dir: Path = Path("./outputs")


def collect_train_arrays(data_root: Path) -> List[np.ndarray]:
    train_dir = data_root / "train"
    paths = list_csvs(train_dir)
    arrays = []
    for p in paths:
        # df, _ = load_csv_with_optional_label(p)
        df, _ = read_skab_csv(p)    
        arr = df.values.astype(np.float32)
        arrays.append(arr)
    return arrays


def compute_train_stats(arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.concatenate(arrays, axis=0)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sd


def build_train_dataset(arrays: List[np.ndarray], mu: np.ndarray, sd: np.ndarray, window: int, stride: int) -> SlidingWindowSeries:
    normed = [ (a - mu) / sd for a in arrays ]
    ds = SlidingWindowSeries(normed, labels=None, window=window, stride=stride, normalize=False, return_labels=False)
    return ds


def train_ssl(cfg: TrainConfig):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    arrays = collect_train_arrays(cfg.data_root)
    assert len(arrays) > 0, "No train CSV files found."

    mu, sd = compute_train_stats(arrays)
    ds = build_train_dataset(arrays, mu, sd, cfg.window, cfg.stride)
    loader = DataLoader(ds, batch_size=cfg.batch, shuffle=True, num_workers=4, drop_last=True)

    in_feat = ds.X.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TS2VecModel(in_feat, out_dim=256).to(device)
    aug = TimeSeriesAug()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_loss = 1e9
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for x in loader:
            x = x.to(device)  # [B, L, F]
            x1 = aug(x)
            x2 = aug(x)
            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent(z1, z2, temp=cfg.temp)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        avg = float(np.mean(losses)) if losses else 0.0
        print(f"epoch {epoch:03d} | loss {avg:.4f}")
        # save checkpoint
        ckpt = {
            'state_dict': model.state_dict(),
            'mu': mu, 'sd': sd,
            'config': cfg.__dict__
        }
        torch.save(ckpt, cfg.out_dir / "ts2vec_ckpt.pt")
        if avg < best_loss:
            best_loss = avg
            torch.save(ckpt, cfg.out_dir / "ts2vec_best.pt")
    print("Training finished. Best loss:", best_loss)


# ---------------------------------------------------------
# Memory bank (train embeddings) and scoring heads
# ---------------------------------------------------------
@dataclass
class BankConfig:
    data_root: Path
    window: int = 256
    stride: int = 64
    ckpt: Path = Path("./outputs/ts2vec_best.pt")
    out_dir: Path = Path("./outputs")
    k_neighbors: int = 10


def load_model_and_stats(ckpt_path: Path, in_feat: Optional[int] = None) -> Tuple[TS2VecModel, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ckpt = torch.load(ckpt_path, map_location=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # if in_feat is None:
    #     # will infer later
    #     in_feat = ckpt.get('in_feat', None)

    # ★ 추가: state_dict 꺼내기
    state = ckpt['state_dict']
    
    # ★ 체크포인트에서 기대 입력채널 수 추출
    expected_in = state['encoder.net.0.conv.weight'].shape[1]

    # model = TS2VecModel(in_feat if in_feat is not None else 1, out_dim=256).to(device)
    # ★ 중요: in_feat 무시하고 expected_in으로 모델 생성
    model = TS2VecModel(expected_in, out_dim=256).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    
    mu = ckpt['mu']; sd = ckpt['sd']
    # return model, mu, sd
    return model, mu, sd, expected_in


def embed_windows(model: TS2VecModel, X: np.ndarray, batch: int = 256) -> np.ndarray:
    device = next(model.parameters()).device
    zs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(device)
            z = model(xb).cpu().numpy()
            zs.append(z)
    return np.concatenate(zs, axis=0)


def build_memory_bank(cfg: BankConfig):
    ensure_dir(cfg.out_dir)
    arrays = collect_train_arrays(cfg.data_root)
    mu, sd = compute_train_stats(arrays)
    ds = build_train_dataset(arrays, mu, sd, cfg.window, cfg.stride)
    in_feat = ds.X.shape[-1]

    model, _, _ = load_model_and_stats(cfg.ckpt, in_feat=in_feat)
    Z = embed_windows(model, ds.X)

    # kNN index
    knn = NearestNeighbors(n_neighbors=cfg.k_neighbors, algorithm='auto')
    knn.fit(Z)

    # KDE density
    kde = KernelDensity(kernel='gaussian', bandwidth='scott')
    # sklearn doesn't support 'scott' directly; compute bandwidth via rule-of-thumb
    # Here, approximate Scott's rule: bw = n^{-1/(d+4)} * std
    n, d = Z.shape
    std = Z.std(axis=0).mean() + 1e-8
    bw = (n ** (-1.0 / (d + 4))) * std
    kde = KernelDensity(kernel='gaussian', bandwidth=max(bw, 1e-3))
    kde.fit(Z)

    bank = {
        'Z': Z.astype(np.float32),
        'knn': knn,
        'k_neighbors': cfg.k_neighbors,
        'kde_bw': float(max(bw, 1e-3)),
        'mu': mu, 'sd': sd,
    }
    # save with joblib-like manual serialize
    import pickle
    with open(cfg.out_dir / 'memory_knn.pkl', 'wb') as f:
        pickle.dump(knn, f)
    with open(cfg.out_dir / 'memory_kde.pkl', 'wb') as f:
        pickle.dump(kde, f)
    np.save(cfg.out_dir / 'memory_Z.npy', Z.astype(np.float32))
    np.save(cfg.out_dir / 'stats_mu.npy', mu)
    np.save(cfg.out_dir / 'stats_sd.npy', sd)
    print(f"Memory bank built: Z={Z.shape}, k={cfg.k_neighbors}, KDE bw={max(bw,1e-3):.4f}")


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
@dataclass
class EvalConfig:
    data_root: Path
    window: int = 256
    stride: int = 64
    ckpt: Path = Path("./outputs/ts2vec_best.pt")
    bank_dir: Path = Path("./outputs")
    score_head: str = 'kde'  # 'kde' or 'knn'
    smooth: int = 5
    out_dir: Path = Path("./outputs")


def collect_test_arrays(data_root: Path) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    test_dir = data_root / "test"
    paths = list_csvs(test_dir)
    Xs, Ys = [], []
    found_label = False
    for p in paths:
        # df, y = load_csv_with_optional_label(p)
        df, y = read_skab_csv(p)   
        Xs.append(df.values.astype(np.float32))
        if y is not None:
            found_label = True
            Ys.append(y.astype(np.int64))
        else:
            Ys.append(None)
    if not found_label:
        return Xs, None
    return Xs, Ys


def kde_crossing_threshold(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    # estimate densities and find first intersection
    from sklearn.neighbors import KernelDensity
    xs = np.linspace(min(scores_pos.min(), scores_neg.min()), max(scores_pos.max(), scores_neg.max()), 512)[:,None]
    def fit_kde(x):
        x = x.reshape(-1,1)
        std = x.std() + 1e-8
        bw = (len(x) ** (-1.0 / 5)) * std  # d=1
        kde = KernelDensity(kernel='gaussian', bandwidth=max(bw,1e-3)).fit(x)
        return kde
    kde_p = fit_kde(scores_pos)
    kde_n = fit_kde(scores_neg)
    log_p = kde_p.score_samples(xs)
    log_n = kde_n.score_samples(xs)
    diff = (log_p - log_n)
    idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    thr = xs[idx[0],0] if len(idx)>0 else float(np.median(scores_pos))
    return float(thr)


def align_array_cols(arr: np.ndarray, expected_in: int) -> np.ndarray:
    F = arr.shape[1]
    if F == expected_in:
        return arr
    elif F > expected_in:  # 초과 컬럼 자르기
        return arr[:, :expected_in]
    else:                  # 부족하면 0 padding
        pad = np.zeros((arr.shape[0], expected_in - F), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=1)



def eval_anomaly(cfg: EvalConfig):
    ensure_dir(cfg.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load stats & bank
    mu = np.load(cfg.bank_dir / 'stats_mu.npy')
    sd = np.load(cfg.bank_dir / 'stats_sd.npy')
    Zbank = np.load(cfg.bank_dir / 'memory_Z.npy')
    import pickle
    with open(cfg.bank_dir / 'memory_knn.pkl','rb') as f:
        knn = pickle.load(f)
    with open(cfg.bank_dir / 'memory_kde.pkl','rb') as f:
        kde = pickle.load(f)

    # collect test
    Xs, Ys = collect_test_arrays(cfg.data_root)
    assert len(Xs) > 0, "No test CSV files found."

    # prepare model
    in_feat = Xs[0].shape[1]
    # model, _, _ = load_model_and_stats(cfg.ckpt, in_feat=in_feat)
    model, _, _, expected_in = load_model_and_stats(cfg.ckpt, in_feat=in_feat)


    all_scores = []
    all_labels = [] if Ys is not None else None

    for i, arr in enumerate(Xs):
         # ★ 1) 먼저 채널 정렬(학습 expected_in에 맞춤)
        arr = align_array_cols(arr, expected_in)

        # ★ 2) 그 다음 정규화 (mu, sd는 (1, expected_in) 형태)
        # windowing
        arrn = (arr - mu) / sd
        ds = SlidingWindowSeries([arrn], labels=[Ys[i]] if Ys is not None else None,
                                 window=cfg.window, stride=cfg.stride, normalize=False, return_labels=(Ys is not None))
        Xw = ds.X  # [Nw, L, F]

        # ★ 여기서 특성 수 정렬
        Xw = align_features(Xw, expected_in)

        Zw = embed_windows(model, Xw)

        if cfg.score_head == 'knn':
            dists, _ = knn.kneighbors(Zw, n_neighbors=knn.n_neighbors, return_distance=True)
            scores = dists.mean(axis=1)
        else:
            logp = kde.score_samples(Zw)
            scores = -logp
        # smooth over time windows (per file)
        scores = moving_average(scores, cfg.smooth)
        all_scores.append(scores)
        if Ys is not None:
            all_labels.append(ds.Y)

    # concatenate
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels) if all_labels is not None else None

    # choose threshold via KDE crossing using a small validation slice
    metrics = {}
    if labels is not None:
        pos = scores[labels==1]
        neg = scores[labels==0]
        thr = kde_crossing_threshold(pos, neg)
        preds = (scores >= thr).astype(int)
        metrics['AUC_ROC'] = roc_auc_score(labels, scores)
        metrics['AUC_PR' ] = average_precision_score(labels, scores)
        # best F1 for reference
        ps, rs, ths = precision_recall_curve(labels, scores)
        f1s = 2*ps*rs/(ps+rs+1e-8)
        metrics['F1_best'] = float(np.max(f1s))
        metrics['Threshold_KDE_cross'] = float(thr)
        print(json.dumps(metrics, indent=2))
        # save per-window results
        out_csv = Path(cfg.out_dir) / 'test_scores.csv'
        pd.DataFrame({'score':scores, 'label':labels, 'pred':preds}).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
    else:
        out_csv = Path(cfg.out_dir) / 'test_scores_unlabeled.csv'
        pd.DataFrame({'score':scores}).to_csv(out_csv, index=False)
        print(f"Saved (unlabeled): {out_csv}")


# ---------------------------------------------------------
# Feature export (CSV + JSON)
# ---------------------------------------------------------
def export_features(data_root: Path, out_dir: Path, splits=('train','test'), json_orient='records'):
    """
    Read every CSV under data_root/{split}/*.csv using read_skab_csv(),
    then save numeric feature table to:
      - {out_dir}/{split}/{stem}_features.csv
      - {out_dir}/{split}/{stem}_features.json
    json_orient: 'records' (row-wise list of dicts) or 'split' (columns->lists)
    """
    ensure_dir(out_dir)
    for split in splits:
        split_dir = data_root / split
        paths = list_csvs(split_dir)
        if not paths:
            print(f"[export] no files under: {split_dir}")
            continue
        out_split = out_dir / split
        ensure_dir(out_split)

        for p in paths:
            df, y = read_skab_csv(p)  # 숫자형만, 라벨/시간컬럼 제거된 feature 테이블
            stem = p.stem

            # CSV 저장
            csv_path = out_split / f"{stem}_features.csv"
            df.to_csv(csv_path, index=False)

            # JSON 저장
            json_path = out_split / f"{stem}_features.json"
            # orient='records'이면 [{"f1":..,"f2":..}, ...]
            # orient='split'이면 {"index":[...],"columns":[...],"data":[[...],...]}
            df.to_json(json_path, orient=json_orient)

            # (옵션) 메타: feature 이름 저장
            # meta_path = out_split / f"{stem}_feature_names.json"
            # with open(meta_path, "w") as f:
            #     json.dump({"features": list(df.columns)}, f, ensure_ascii=False, indent=2)
            
            # 피처 이름은 split별로 한 번만 저장
            meta_path = out_split / "feature_names.json"
            if not meta_path.exists():
                with open(meta_path, "w") as f:
                    json.dump({"features": list(df.columns)}, f, ensure_ascii=False, indent=2)


            print(f"[export] {split}/{p.name} -> {csv_path.name}, {json_path.name}")



# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('train')
    p_train.add_argument('--data_root', type=Path, required=True)
    p_train.add_argument('--window', type=int, default=256)
    p_train.add_argument('--stride', type=int, default=64)
    p_train.add_argument('--batch', type=int, default=128)
    p_train.add_argument('--epochs', type=int, default=80)
    p_train.add_argument('--lr', type=float, default=1e-3)
    p_train.add_argument('--wd', type=float, default=1e-4)
    p_train.add_argument('--temp', type=float, default=0.1)
    p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--out_dir', type=Path, default=Path('./outputs'))

    p_bank = sub.add_parser('build_bank')
    p_bank.add_argument('--data_root', type=Path, required=True)
    p_bank.add_argument('--window', type=int, default=256)
    p_bank.add_argument('--stride', type=int, default=64)
    p_bank.add_argument('--ckpt', type=Path, default=Path('./outputs/ts2vec_best.pt'))
    p_bank.add_argument('--out_dir', type=Path, default=Path('./outputs'))
    p_bank.add_argument('--k_neighbors', type=int, default=10)

    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--data_root', type=Path, required=True)
    p_eval.add_argument('--window', type=int, default=256)
    p_eval.add_argument('--stride', type=int, default=64)
    p_eval.add_argument('--ckpt', type=Path, default=Path('./outputs/ts2vec_best.pt'))
    p_eval.add_argument('--bank_dir', type=Path, default=Path('./outputs'))
    p_eval.add_argument('--score_head', type=str, default='kde', choices=['kde','knn'])
    p_eval.add_argument('--smooth', type=int, default=5)
    p_eval.add_argument('--out_dir', type=Path, default=Path('./outputs'))


    p_dump = sub.add_parser('export_features')
    p_dump.add_argument('--data_root', type=Path, required=True)
    p_dump.add_argument('--out_dir', type=Path, default=Path('./feature_dumps'))
    p_dump.add_argument('--splits', nargs='+', default=['train','test'], choices=['train','test'])
    p_dump.add_argument('--json_orient', type=str, default='records', choices=['records','split'])

    # args = parser.parse_args()
    # cmd = getattr(args, "cmd", None)

    # if hasattr(args, "cmd"):
    #     delattr(args, "cmd")

    # if args.cmd == 'train':
    #     cfg = TrainConfig(**vars(args))
    #     train_ssl(cfg)
    # elif args.cmd == 'build_bank':
    #     cfg = BankConfig(**vars(args))
    #     build_memory_bank(cfg)
    # elif args.cmd == 'eval':
    #     cfg = EvalConfig(**vars(args))
    #     eval_anomaly(cfg)

    args = parser.parse_args()

    cmd = getattr(args, "cmd", None)   # ← 먼저 꺼내두고
    if hasattr(args, "cmd"):
        delattr(args, "cmd")           # ← dataclass에 안 들어가게 제거

    # ↓↓↓ 여기서부터는 args.cmd 대신 cmd 사용
    if cmd == 'train':
        cfg = TrainConfig(**vars(args))
        train_ssl(cfg)
    elif cmd == 'build_bank':
        cfg = BankConfig(**vars(args))
        build_memory_bank(cfg)
    elif cmd == 'eval':
        cfg = EvalConfig(**vars(args))
        eval_anomaly(cfg)
    elif cmd == 'export_features':
        export_features(args.data_root, args.out_dir, tuple(args.splits), args.json_orient)
    else:
        raise ValueError("cmd must be one of: train, build_bank, eval")


    # args = parser.parse_args()
    # ad = vars(args).copy()
    # cmd = ad.pop('cmd', None)  # ← 키를 꺼내면서 dict에서 제거

    # if cmd == 'train':
    #     cfg = TrainConfig(**ad)
    #     train_ssl(cfg)
    # elif cmd == 'build_bank':
    #     cfg = BankConfig(**ad)
    #     build_memory_bank(cfg)
    # elif cmd == 'eval':
    #     cfg = EvalConfig(**ad)
    #     eval_anomaly(cfg)
    # else:
    #     raise ValueError("cmd must be one of: train, build_bank, eval")


if __name__ == '__main__':
    main()
