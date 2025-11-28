

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
# 유틸리티 함수
# ---------------------------------------------------------

def set_seed(seed: int = 42):
    """재현성을 위한 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    """디렉토리가 없으면 생성 (부모 디렉토리 포함)"""
    p.mkdir(parents=True, exist_ok=True)


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """이동 평균 스무딩 (이상 점수를 부드럽게 만들기 위해 사용)
    
    Args:
        x: 입력 배열
        k: 윈도우 크기
    Returns:
        스무딩된 배열 (원본과 동일한 길이)
    """
    if k <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    out = (cumsum[k:] - cumsum[:-k]) / float(k)
    # 앞부분을 첫 값으로 패딩하여 길이 맞춤
    pad = np.concatenate([np.repeat(out[0], k-1), out])
    return pad[: len(x)]


# ---------------------------------------------------------
# 데이터 처리: SKAB 데이터셋
# ---------------------------------------------------------

# 라벨 및 시간 컬럼명 후보 (여러 형식 지원)
LABEL_CANDS = ['anomaly','label','y','class','Class','is_anomaly']
TIME_CANDS  = ['timestamp','time','datetime','date','index']

def read_skab_csv(path):
    """SKAB CSV 파일을 읽고 특성과 라벨을 분리
    
    처리 과정:
    1. 구분자 자동 추정 (쉼표, 세미콜론 등)
    2. Unnamed 컬럼 제거 (인덱스 중복 등)
    3. 라벨 컬럼 추출 및 제거 (특성과 분리)
    4. 시간 컬럼 제거 (특성으로 사용 안 함)
    5. 숫자형 컬럼만 유지
    
    Args:
        path: CSV 파일 경로
    Returns:
        df: 숫자형 특성만 포함한 DataFrame
        y: 라벨 배열 (없으면 None)
    """
    # 1) 구분자 자동 추정
    df = pd.read_csv(path, sep=None, engine='python')
    # 2) 한 컬럼으로 읽혔으면 세미콜론으로 재시도
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=';', engine='python')

    # Unnamed 컬럼 제거 (pandas가 자동 생성한 인덱스 컬럼 등)
    drop_unnamed = [c for c in df.columns if str(c).startswith('Unnamed')]
    if drop_unnamed:
        df = df.drop(columns=drop_unnamed)
        
    # 3) 라벨 컬럼 찾아서 분리
    y = None
    for c in LABEL_CANDS:
        if c in df.columns:
            y = df[c].astype(int).values
            df = df.drop(columns=[c])
            break
    
    # 4) 시간 컬럼 제거 (특성으로 사용하지 않음)
    for c in TIME_CANDS:
        if c in df.columns:
            df = df.drop(columns=[c])
    
    # 5) 숫자형 컬럼만 유지 (특성으로 사용)
    df = df.select_dtypes(include=[np.number])
    return df, y



LABEL_CANDIDATES = ["anomaly", "label", "y"]

def load_csv_with_optional_label(path: Path) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """CSV 로드 및 라벨 분리 (간소화 버전, 현재 미사용)"""
    df = pd.read_csv(path)
    label = None
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            label = df[c].astype(int).values
            df = df.drop(columns=[c])
            break
    # 시간 컬럼 제거
    for c in ["timestamp", "time", "datetime", "date"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df, label


def list_csvs(folder: Path) -> List[Path]:
    """폴더 내 모든 CSV 파일 경로를 정렬하여 반환"""
    return sorted([p for p in folder.glob("*.csv") if p.is_file()])

def align_features(X: np.ndarray, expected_in: int) -> np.ndarray:
    
    F = X.shape[-1]
    if F == expected_in:
        return X
    elif F > expected_in:
        return X[..., :expected_in]  # 초과 컬럼 잘라내기
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
                 pos_ratio_threshold: float = 0.5,
                 use_center_label: bool = False):
        """
        Args:
            arrays: [T, F] 형태의 시계열 배열 리스트 (파일별)
            labels: [T] 형태의 라벨 배열 리스트 (0/1) 또는 None
            window: 윈도우 길이 (시간 스텝 수)
            stride: 윈도우 이동 간격 (겹침 정도 조절)
            normalize: True면 전역 정규화 수행 (평균=0, 표준편차=1)
            return_labels: True면 각 윈도우의 라벨 반환
            pos_ratio_threshold: 윈도우를 이상으로 판단하는 임계값 (라벨 평균)
            use_center_label: True면 윈도우 중간 포인트의 라벨만 사용 (F1-PA용)
        
        예: threshold=0.5이면 윈도우의 50% 이상이 이상치일 때 이상 윈도우로 분류
        """
        self.window = window
        self.stride = stride
        self.return_labels = return_labels and (labels is not None)
        self.use_center_label = use_center_label
        Xs, Ys = [], []
        
        # 각 파일별로 슬라이딩 윈도우 생성
        for i, arr in enumerate(arrays):
            L = arr.shape[0]
            y = labels[i] if labels is not None else None
            
            # stride 간격으로 윈도우 추출
            for s in range(0, max(L - window + 1, 0), stride):
                seg = arr[s: s + window]  # [window, F]
                Xs.append(seg)
                
                if self.return_labels:
                    if use_center_label:
                        # 윈도우 중간 포인트의 라벨만 사용 (예: 윈도우 크기 32면 인덱스 15)
                        center_idx = s + window // 2
                        yflag = int(y[center_idx])
                    else:
                        ywin = y[s: s + window]
                        # 윈도우 내 이상치 비율이 임계값 초과하면 이상 윈도우
                        yflag = 1 if (ywin.mean() > pos_ratio_threshold) else 0
                    Ys.append(yflag)
        
        # 리스트를 배열로 변환
        self.X = np.stack(Xs, axis=0).astype(np.float32) if Xs else np.zeros((0, window, arrays[0].shape[1]), dtype=np.float32)
        self.Y = np.array(Ys, dtype=np.int64) if self.return_labels and Ys else None
        
        # 전역 정규화 (주로 테스트용, 학습용은 외부에서 미리 정규화)
        if normalize:
            mu = self.X.reshape(-1, self.X.shape[-1]).mean(axis=0, keepdims=True)
            sd = self.X.reshape(-1, self.X.shape[-1]).std(axis=0, keepdims=True) + 1e-8
            self.X = (self.X - mu) / sd

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # [window, F]
        if self.return_labels:
            return x, self.Y[idx]
        else:
            return x


# ---------------------------------------------------------
# 데이터 증강: TS2Vec 스타일 자기지도학습용
# ---------------------------------------------------------
class TimeSeriesAug:
    """시계열 데이터 증강 (Contrastive Learning용)
    
    다양한 증강 기법을 확률적으로 적용하여 positive pair 생성
    - Jitter: 노이즈 추가
    - Scaling: 크기 조정
    - Time Mask: 시간축 마스킹
    - Crop: 랜덤 자르기 및 패딩
    """
    def __init__(self, jitter_p=0.3, scaling_p=0.2, time_mask_p=0.2, crop_p=0.3):
        """
        Args:
            jitter_p: 노이즈 추가 확률
            scaling_p: 스케일링 확률
            time_mask_p: 시간 마스킹 확률
            crop_p: 랜덤 크롭 확률
        """
        self.jitter_p = jitter_p
        self.scaling_p = scaling_p
        self.time_mask_p = time_mask_p
        self.crop_p = crop_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C] 형태의 시계열 배치 (B=배치, L=길이, C=채널)
        Returns:
            증강된 시계열 [B, L, C]
        """
        B, L, C = x.shape
        out = x.clone()
        device = x.device
        
        # 1) Jitter: 작은 가우시안 노이즈 추가
        if random.random() < self.jitter_p:
            noise = torch.randn_like(out) * 0.01
            out = out + noise
        
        # 2) Scaling: 0.9~1.1 범위로 스케일 조정
        if random.random() < self.scaling_p:
            scale = (0.9 + 0.2 * torch.rand(B, 1, C, device=device))
            out = out * scale
        
        # 3) Time Mask: 시간축의 10% 구간을 0으로 마스킹
        if random.random() < self.time_mask_p:
            mask_len = max(1, int(L * 0.1))
            start = random.randint(0, max(L - mask_len, 0))
            out[:, start:start+mask_len, :] = 0.0
        
        # 4) Random Crop: 60~100% 길이로 자르고 패딩으로 원래 길이 복원
        if random.random() < self.crop_p:
            keep = random.randint(int(L*0.6), L)
            start = random.randint(0, L - keep)
            cropped = out[:, start:start+keep, :]
            # 랜덤한 위치에 패딩
            pad_left = random.randint(0, L - keep)
            pad_right = L - keep - pad_left
            out = F.pad(cropped, (0,0, pad_left, pad_right))  # F는 torch.nn.functional
        
        return out


# ---------------------------------------------------------
# 모델 아키텍처: TS2Vec 스타일 인코더
# ---------------------------------------------------------
class TCNBlock(nn.Module):
    """Temporal Convolutional Network 블록 (residual connection 포함)"""
    def __init__(self, in_ch, out_ch, k=5, d=1):
        """
        Args:
            in_ch: 입력 채널 수
            out_ch: 출력 채널 수
            k: 커널 크기
            d: dilation rate (시간축 수용 범위 확장)
        """
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=d*(k-1)//2, dilation=d)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.res = (in_ch == out_ch)  # 채널 수가 같을 때만 residual connection
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # x: [B, C, L]
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        y = self.dropout(y)
        if self.res:
            y = y + x  # residual connection
        return y

class TS2VecEncoder(nn.Module):
    """TS2Vec 스타일의 시계열 인코더
    
    TCN 블록을 여러 층 쌓고, dilation을 증가시켜 넓은 시간 범위 포착
    마지막에 adaptive pooling으로 고정 길이 임베딩 생성
    """
    def __init__(self, in_feat: int, hid: int = 256, depth: int = 4):
        """
        Args:
            in_feat: 입력 특성 개수
            hid: 은닉층 채널 수
            depth: TCN 블록 개수
        """
        super().__init__()
        chs = [in_feat, hid, hid, hid, hid]
        blocks = []
        for i in range(depth):
            # dilation을 지수적으로 증가 (1, 2, 4, ...) → 넓은 수용 범위
            blocks.append(TCNBlock(chs[i], chs[i+1], k=5, d=2**i if i<3 else 1))
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 시간축 평균 풀링

    def forward(self, x):  # x: [B, L, F]
        x = x.transpose(1, 2)  # [B, F, L] - Conv1d는 (B, C, L) 형식
        h = self.net(x)        # [B, C, L]
        g = self.pool(h).squeeze(-1)  # [B, C] - 시간축 평균
        return g

class ProjectionHead(nn.Module):
    """Contrastive Learning용 Projection Head
    
    임베딩을 저차원 공간으로 매핑하고 L2 정규화
    """
    def __init__(self, in_dim: int, out_dim: int = 256):
        """
        Args:
            in_dim: 입력 차원
            out_dim: 출력 차원 (최종 임베딩 차원)
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        z = self.fc2(self.act(self.fc1(x)))
        z = F.normalize(z, dim=-1)  # L2 정규화 (코사인 유사도 사용 위해)
        return z

class TS2VecModel(nn.Module):
    """전체 TS2Vec 모델 (Encoder + Projection Head)"""
    def __init__(self, in_feat: int, out_dim: int = 256):
        """
        Args:
            in_feat: 입력 특성 개수
            out_dim: 최종 임베딩 차원 (기본 256)
        """
        super().__init__()
        self.encoder = TS2VecEncoder(in_feat)
        self.proj = ProjectionHead(256, out_dim)

    def forward(self, x):
        g = self.encoder(x)
        z = self.proj(g)
        return z  # [B, out_dim]


# ---------------------------------------------------------
# Contrastive Loss: NT-Xent (Normalized Temperature-scaled Cross Entropy)
# ---------------------------------------------------------

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """NT-Xent Loss (SimCLR에서 사용하는 Contrastive Loss)
    
    동일한 샘플의 두 증강 버전(z1, z2)은 가깝게, 다른 샘플과는 멀게 학습
    
    Args:
        z1: 첫 번째 증강 버전의 임베딩 [B, D]
        z2: 두 번째 증강 버전의 임베딩 [B, D]
        temp: temperature 파라미터 (작을수록 hard negative에 집중)
    Returns:
        contrastive loss
    """
    # L2 정규화 (코사인 유사도 계산을 위해)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B, D = z1.shape
    
    # 두 증강 버전을 하나로 합침 [2B, D]
    z = torch.cat([z1, z2], dim=0)
    
    # 모든 쌍 간의 유사도 계산 [2B, 2B]
    sim = torch.mm(z, z.t())
    
    # 자기 자신과의 유사도는 제외 (마스킹)
    mask = torch.eye(2*B, device=z.device).bool()
    sim = sim / temp  # temperature scaling
    sim.masked_fill_(mask, -1e9)
    
    # positive pair 정의: (i, i+B), (i+B, i)
    # 즉, 같은 샘플의 두 증강 버전끼리만 positive
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)
    
    # cross entropy로 loss 계산 (positive와 가깝게, negative와 멀게)
    loss = F.cross_entropy(sim, pos)
    return loss


# ---------------------------------------------------------
# 자기지도학습 (Self-Supervised Learning) - 정상 데이터만 사용
# ---------------------------------------------------------
@dataclass
class TrainConfig:
    """학습 설정 파라미터"""
    data_root: Path      # 데이터 루트 경로 (train/test 폴더 포함)
    window: int = 32     # 슬라이딩 윈도우 길이
    stride: int = 1      # 윈도우 이동 간격
    batch: int = 128     # 배치 크기
    epochs: int = 80     # 학습 에포크 수
    lr: float = 1e-3     # 학습률
    wd: float = 1e-4     # weight decay (L2 정규화)
    temp: float = 0.1    # contrastive loss의 temperature
    seed: int = 42       # 랜덤 시드
    out_dir: Path = Path("./outputs")  # 출력 디렉토리 (체크포인트 저장)


def collect_train_arrays(data_root: Path) -> List[np.ndarray]:
    """학습 폴더의 모든 CSV를 읽어 배열 리스트 반환
    
    Args:
        data_root: 데이터 루트 경로
    Returns:
        [T, F] 형태의 배열 리스트 (파일별)
    """
    # train 또는 anomaly-free 폴더 모두 지원
    if (data_root / "anomaly-free").exists():
        train_dir = data_root / "anomaly-free"
    else:
        train_dir = data_root / "train"
    
    paths = list_csvs(train_dir)
    arrays = []
    for p in paths:
        df, _ = read_skab_csv(p)  # 라벨은 무시 (정상 데이터 가정)
        arr = df.values.astype(np.float32)
        arrays.append(arr)
    return arrays


def compute_train_stats(arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """학습 데이터의 평균/표준편차 계산 (정규화용)
    
    모든 파일을 통합하여 전역 통계량 계산
    
    Args:
        arrays: 시계열 배열 리스트
    Returns:
        mu: 평균 [1, F]
        sd: 표준편차 [1, F]
    """
    X = np.concatenate(arrays, axis=0)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8  # 0으로 나누기 방지
    return mu, sd


def build_train_dataset(arrays: List[np.ndarray], mu: np.ndarray, sd: np.ndarray, window: int, stride: int) -> SlidingWindowSeries:
    """정규화 후 슬라이딩 윈도우 데이터셋 생성
    
    Args:
        arrays: 시계열 배열 리스트
        mu: 평균
        sd: 표준편차
        window: 윈도우 길이
        stride: 윈도우 간격
    Returns:
        학습용 Dataset
    """
    # 각 파일을 학습 통계로 정규화
    normed = [ (a - mu) / sd for a in arrays ]
    # 슬라이딩 윈도우 생성 (라벨 없음, 추가 정규화 불필요)
    ds = SlidingWindowSeries(normed, labels=None, window=window, stride=stride, normalize=False, return_labels=False)
    return ds


def train_ssl(cfg: TrainConfig):
    """자기지도학습으로 TS2Vec 인코더 학습
    
    학습 과정:
    1. 정상 데이터 로드 및 정규화
    2. 슬라이딩 윈도우 생성
    3. Contrastive Learning: 같은 윈도우의 두 증강 버전은 가깝게, 다른 윈도우와는 멀게
    4. 매 에포크마다 체크포인트 저장 (최고 성능 모델 별도 저장)
    
    Args:
        cfg: 학습 설정
    """
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    # 1) 학습 데이터 로드
    arrays = collect_train_arrays(cfg.data_root)
    assert len(arrays) > 0, "No train CSV files found."

    # 2) 정규화 통계 계산 및 데이터셋 생성
    mu, sd = compute_train_stats(arrays)
    ds = build_train_dataset(arrays, mu, sd, cfg.window, cfg.stride)
    loader = DataLoader(ds, batch_size=cfg.batch, shuffle=True, num_workers=4, drop_last=True)

    # 3) 모델 및 증강 초기화
    in_feat = ds.X.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TS2VecModel(in_feat, out_dim=256).to(device)
    aug = TimeSeriesAug()

    # 4) 옵티마이저 및 스케줄러
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    # 5) 학습 루프
    best_loss = 1e9
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for x in loader:
            x = x.to(device)  # [B, L, F]
            
            # 같은 윈도우에 두 가지 다른 증강 적용
            x1 = aug(x)
            x2 = aug(x)
            
            # 임베딩 추출
            z1 = model(x1)
            z2 = model(x2)
            
            # Contrastive Loss: z1, z2는 가깝게, 다른 샘플과는 멀게
            loss = nt_xent(z1, z2, temp=cfg.temp)
            
            # 역전파 및 가중치 업데이트
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            opt.step()
            losses.append(loss.item())
        
        sched.step()  # 학습률 조정
        avg = float(np.mean(losses)) if losses else 0.0
        print(f"epoch {epoch:03d} | loss {avg:.4f}")
        
        # 6) 체크포인트 저장 (정규화 통계 포함)
        ckpt = {
            'state_dict': model.state_dict(),
            'mu': mu, 'sd': sd,  # 추론 시 필요
            'config': cfg.__dict__
        }
        torch.save(ckpt, cfg.out_dir / "ts2vec_ckpt.pt")
        
        # 최고 성능 모델 별도 저장
        if avg < best_loss:
            best_loss = avg
            torch.save(ckpt, cfg.out_dir / "ts2vec_best.pt")
    
    print("Training finished. Best loss:", best_loss)


# ---------------------------------------------------------
# 메모리 뱅크: 정상 데이터의 임베딩으로 이상 탐지 기준 구축
# ---------------------------------------------------------
@dataclass
class BankConfig:
    """메모리 뱅크 구축 설정"""
    data_root: Path
    window: int = 32
    stride: int = 1
    ckpt: Path = Path("./outputs/ts2vec_best.pt")  # 학습된 모델 경로
    out_dir: Path = Path("./outputs")
    k_neighbors: int = 10  # kNN에서 사용할 이웃 개수


def load_model_and_stats(ckpt_path: Path, in_feat: Optional[int] = None):
    """체크포인트에서 모델 및 정규화 통계 로드
    
    ⚠️ 중요: 
    - 체크포인트에서 입력 특성 개수를 자동으로 추출 (학습 시 사용한 특성 개수)
    - in_feat 파라미터는 무시됨 (하위 호환성 유지용)
    
    Args:
        ckpt_path: 체크포인트 파일 경로
        in_feat: (사용 안 함, 하위 호환성용)
    Returns:
        model: 로드된 모델 (eval 모드)
        mu: 정규화 평균
        sd: 정규화 표준편차
        expected_in: 학습 시 사용한 입력 특성 개수
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # state_dict에서 첫 번째 conv 레이어의 입력 채널 수 추출
    state = ckpt['state_dict']
    expected_in = state['encoder.net.0.conv.weight'].shape[1]

    # 학습 시 사용한 특성 개수로 모델 생성
    model = TS2VecModel(expected_in, out_dim=256).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    
    mu = ckpt['mu']
    sd = ckpt['sd']
    return model, mu, sd, expected_in


def embed_windows(model: TS2VecModel, X: np.ndarray, batch: int = 256) -> np.ndarray:
    """윈도우 배열을 임베딩으로 변환
    
    배치 단위로 처리하여 메모리 효율성 확보
    
    Args:
        model: 학습된 TS2Vec 모델
        X: [N, L, F] 형태의 윈도우 배열
        batch: 배치 크기
    Returns:
        Z: [N, D] 형태의 임베딩 배열 (D=256)
    """
    device = next(model.parameters()).device
    zs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(device)
            z = model(xb).cpu().numpy()
            zs.append(z)
    return np.concatenate(zs, axis=0)


def build_memory_bank(cfg: BankConfig):
    """정상 데이터의 임베딩으로 메모리 뱅크 구축
    
    과정:
    1. 학습 데이터를 윈도우로 분할 및 정규화
    2. 학습된 모델로 임베딩 추출
    3. kNN 인덱스 구축 (거리 기반 이상 점수)
    4. KDE 밀도 추정 (밀도 기반 이상 점수)
    5. 모든 정보 저장 (임베딩, kNN, KDE, 정규화 통계)
    
    ⚠️ 버그 수정: load_model_and_stats는 4개 값 반환 (model, mu, sd, expected_in)
    
    Args:
        cfg: 메모리 뱅크 구축 설정
    """
    ensure_dir(cfg.out_dir)
    arrays = collect_train_arrays(cfg.data_root)
    mu, sd = compute_train_stats(arrays)
    ds = build_train_dataset(arrays, mu, sd, cfg.window, cfg.stride)
    in_feat = ds.X.shape[-1]

    # ⚠️ 버그 수정: 4개 값 반환받아야 함
    model, _, _, expected_in = load_model_and_stats(cfg.ckpt, in_feat=in_feat)
    Z = embed_windows(model, ds.X)

    # 1) kNN 인덱스 구축
    # 테스트 임베딩과 정상 임베딩 간 거리 계산용
    knn = NearestNeighbors(n_neighbors=cfg.k_neighbors, algorithm='auto')
    knn.fit(Z)

    # 2) KDE 밀도 추정
    # Scott's rule로 bandwidth 계산: bw = n^{-1/(d+4)} * std
    n, d = Z.shape
    std = Z.std(axis=0).mean() + 1e-8
    bw = (n ** (-1.0 / (d + 4))) * std
    kde = KernelDensity(kernel='gaussian', bandwidth=max(bw, 1e-3))
    kde.fit(Z)

    # 3) 모든 정보 저장
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
# 평가: 테스트 데이터에 대한 이상 점수 계산 및 메트릭 산출
# ---------------------------------------------------------
@dataclass
class EvalConfig:
    """평가 설정 파라미터"""
    data_root: Path
    window: int = 32
    stride: int = 1
    ckpt: Path = Path("./outputs/ts2vec_best.pt")  # 학습된 모델
    bank_dir: Path = Path("./outputs")  # 메모리 뱅크 경로
    score_head: str = 'kde'  # 이상 점수 계산 방법: 'kde' 또는 'knn'
    smooth: int = 5  # 이동 평균 윈도우 크기 (점수 스무딩)
    out_dir: Path = Path("./outputs")  # 결과 저장 경로


def collect_test_arrays(data_root: Path) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """테스트 폴더의 모든 CSV를 읽어 특성과 라벨 반환
    
    3개 그룹(other, valve1, valve2)을 모두 탐색하여 데이터를 수집합니다.
    
    Args:
        data_root: 데이터 루트 경로
    Returns:
        Xs: [T, F] 형태의 특성 배열 리스트
        Ys: [T] 형태의 라벨 배열 리스트 (없으면 None)
    """
    # 3개 그룹 탐색: other, valve1, valve2
    test_groups = []
    for group_name in ['other', 'valve1', 'valve2']:
        group_dir = data_root / group_name
        if group_dir.exists():
            test_groups.append((group_name, group_dir))
    
    # 그룹이 없으면 기본 test 폴더 사용
    if not test_groups:
        if (data_root / "test").exists():
            test_groups = [('test', data_root / "test")]
        else:
            test_groups = [('other', data_root / "other")]
    
    # 모든 그룹의 파일 수집
    Xs, Ys = [], []
    found_label = False
    for group_name, group_dir in test_groups:
        paths = list_csvs(group_dir)
        print(f"Loading {len(paths)} files from {group_name}/")
        for p in paths:
            df, y = read_skab_csv(p)   
            Xs.append(df.values.astype(np.float32))
            if y is not None:
                found_label = True
                Ys.append(y.astype(np.int64))
            else:
                Ys.append(None)
    
    # 모든 파일에 라벨이 없으면 None 반환
    if not found_label:
        return Xs, None
    return Xs, Ys


def kde_crossing_threshold(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """KDE 교차점 기반 임계값 계산
    
    이상 점수의 분포를 정상/이상으로 나눠 KDE로 추정한 뒤,
    두 분포가 교차하는 첫 번째 지점을 임계값으로 사용
    
    Args:
        scores_pos: 이상 샘플들의 점수
        scores_neg: 정상 샘플들의 점수
    Returns:
        threshold: 이상/정상 구분 임계값
    """
    from sklearn.neighbors import KernelDensity
    
    # 점수 범위를 512개 구간으로 나눔
    xs = np.linspace(min(scores_pos.min(), scores_neg.min()), 
                     max(scores_pos.max(), scores_neg.max()), 512)[:,None]
    
    def fit_kde(x):
        """Scott's rule로 bandwidth 계산하여 KDE 학습"""
        x = x.reshape(-1,1)
        std = x.std() + 1e-8
        bw = (len(x) ** (-1.0 / 5)) * std  # d=1 (1차원)
        kde = KernelDensity(kernel='gaussian', bandwidth=max(bw,1e-3)).fit(x)
        return kde
    
    # 정상/이상 각각 KDE 학습
    kde_p = fit_kde(scores_pos)
    kde_n = fit_kde(scores_neg)
    
    # 밀도 추정
    log_p = kde_p.score_samples(xs)
    log_n = kde_n.score_samples(xs)
    
    # 교차점 찾기 (부호 변화 지점)
    diff = (log_p - log_n)
    idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    
    # 교차점이 있으면 첫 번째 교차점, 없으면 이상 점수의 중앙값 사용
    thr = xs[idx[0],0] if len(idx)>0 else float(np.median(scores_pos))
    return float(thr)


def compute_f1_pa(labels: np.ndarray, preds: np.ndarray) -> float:
    """F1-PA (Point-Adjusted F1) 스코어 계산
    
    이상 구간을 하나의 단위로 취급하여, 구간 내 하나의 포인트라도 맞추면
    해당 구간 전체를 올바르게 탐지한 것으로 간주
    
    Args:
        labels: 실제 라벨 [N] (0=정상, 1=이상)
        preds: 예측 라벨 [N] (0=정상, 1=이상)
    Returns:
        f1_pa: Point-Adjusted F1 스코어
    """
    # 이상 구간 추출: 연속된 1들의 구간
    def get_anomaly_segments(y):
        """연속된 이상 구간의 (시작, 끝) 인덱스 리스트 반환"""
        segments = []
        in_segment = False
        start = 0
        for i in range(len(y)):
            if y[i] == 1 and not in_segment:
                start = i
                in_segment = True
            elif y[i] == 0 and in_segment:
                segments.append((start, i))
                in_segment = False
        if in_segment:
            segments.append((start, len(y)))
        return segments
    
    # 실제 이상 구간과 예측 이상 구간
    true_segments = get_anomaly_segments(labels)
    pred_segments = get_anomaly_segments(preds)
    
    if len(true_segments) == 0:
        # 실제 이상이 없는 경우
        return 1.0 if len(pred_segments) == 0 else 0.0
    
    if len(pred_segments) == 0:
        # 예측 이상이 없는 경우
        return 0.0
    
    # Point-Adjusted TP: 실제 구간마다 예측이 하나라도 겹치면 TP
    tp_pa = 0
    for ts, te in true_segments:
        detected = False
        for ps, pe in pred_segments:
            # 구간이 겹치는지 확인
            if not (pe <= ts or ps >= te):
                detected = True
                break
        if detected:
            tp_pa += 1
    
    # Point-Adjusted FP: 예측 구간 중 실제 구간과 겹치지 않는 것
    fp_pa = 0
    for ps, pe in pred_segments:
        overlaps = False
        for ts, te in true_segments:
            if not (pe <= ts or ps >= te):
                overlaps = True
                break
        if not overlaps:
            fp_pa += 1
    
    # Point-Adjusted FN: 탐지되지 않은 실제 구간 수
    fn_pa = len(true_segments) - tp_pa
    
    # F1-PA 계산
    precision_pa = tp_pa / (tp_pa + fp_pa) if (tp_pa + fp_pa) > 0 else 0.0
    recall_pa = tp_pa / (tp_pa + fn_pa) if (tp_pa + fn_pa) > 0 else 0.0
    f1_pa = 2 * precision_pa * recall_pa / (precision_pa + recall_pa) if (precision_pa + recall_pa) > 0 else 0.0
    
    return f1_pa


def align_array_cols(arr: np.ndarray, expected_in: int) -> np.ndarray:
    """2D 배열의 컬럼 수를 expected_in에 맞춤
    
    학습 시와 테스트 시 특성 개수가 다를 때 사용
    - 초과: 앞에서부터 expected_in개만 사용
    - 부족: 뒤에 0으로 패딩
    
    Args:
        arr: [T, F] 형태의 배열
        expected_in: 기대하는 특성 개수
    Returns:
        [T, expected_in] 형태로 정렬된 배열
    """
    F = arr.shape[1]
    if F == expected_in:
        return arr
    elif F > expected_in:  # 초과 컬럼 자르기
        return arr[:, :expected_in]
    else:                  # 부족하면 0 padding
        pad = np.zeros((arr.shape[0], expected_in - F), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=1)



def eval_anomaly(cfg: EvalConfig):
    """테스트 데이터에 대한 이상 탐지 평가
    
    과정:
    1. 메모리 뱅크 및 학습된 모델 로드
    2. 테스트 데이터를 윈도우로 분할 및 임베딩 추출
    3. 메모리 뱅크와 비교하여 이상 점수 계산 (kNN 또는 KDE)
    4. 이동 평균으로 점수 스무딩
    5. KDE 교차점 기반 임계값 설정
    6. 메트릭 계산 (AUC-ROC, AUC-PR, F1, F1-PA) 및 결과 저장
    7. 임베딩 Z를 .pt 파일로 저장
    
    Args:
        cfg: 평가 설정
    """
    ensure_dir(cfg.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 메모리 뱅크 로드 (정규화 통계, 정상 임베딩, kNN, KDE)
    mu = np.load(cfg.bank_dir / 'stats_mu.npy')
    sd = np.load(cfg.bank_dir / 'stats_sd.npy')
    Zbank = np.load(cfg.bank_dir / 'memory_Z.npy')
    import pickle
    with open(cfg.bank_dir / 'memory_knn.pkl','rb') as f:
        knn = pickle.load(f)
    with open(cfg.bank_dir / 'memory_kde.pkl','rb') as f:
        kde = pickle.load(f)

    # 2) 테스트 데이터 로드
    Xs, Ys = collect_test_arrays(cfg.data_root)
    assert len(Xs) > 0, "No test CSV files found."

    # 3) 모델 로드
    in_feat = Xs[0].shape[1]
    model, _, _, expected_in = load_model_and_stats(cfg.ckpt, in_feat=in_feat)

    # 4) 파일별로 이상 점수 계산 및 윈도우별 개별 임베딩 저장
    all_scores = []
    all_labels = [] if Ys is not None else None
    
    # 테스트 데이터 그룹별 디렉토리 탐색
    test_groups = []
    for group_name in ['other', 'valve1', 'valve2']:
        group_dir = cfg.data_root / group_name
        if group_dir.exists():
            test_groups.append((group_name, group_dir))
    
    # 그룹이 없으면 기본 test 폴더 사용
    if not test_groups:
        if (cfg.data_root / "test").exists():
            test_groups = [('test', cfg.data_root / "test")]
        else:
            test_groups = [('other', cfg.data_root / "other")]
    
    # 전체 파일 경로와 그룹 정보 수집
    test_paths_with_group = []
    for group_name, group_dir in test_groups:
        paths = list_csvs(group_dir)
        for p in paths:
            test_paths_with_group.append((p, group_name))
    
    print(f"Found {len(test_paths_with_group)} test files across {len(test_groups)} groups")

    for i, arr in enumerate(Xs):
        file_path, group_name = test_paths_with_group[i]
        file_stem = file_path.stem
        
        # 그룹별 출력 디렉토리 생성
        group_out_dir = Path(cfg.out_dir) / group_name
        ensure_dir(group_out_dir)
        
        # 4-1) 특성 개수를 학습 시와 맞춤 (잘라내기 또는 패딩)
        arr = align_array_cols(arr, expected_in)

        # 4-2) 정규화 (학습 시 통계 사용)
        arrn = (arr - mu) / sd
        
        # 4-3) 슬라이딩 윈도우 생성 (중간 라벨 사용)
        ds = SlidingWindowSeries([arrn], labels=[Ys[i]] if Ys is not None else None,
                                 window=cfg.window, stride=cfg.stride, 
                                 normalize=False, return_labels=(Ys is not None),
                                 use_center_label=True)
        Xw = ds.X  # [Nw, L, F]

        # 4-4) 특성 차원 정렬 (윈도우 단위)
        Xw = align_features(Xw, expected_in)

        # 4-5) 임베딩 추출
        Zw = embed_windows(model, Xw)
        
        # 4-5-1) 각 윈도우를 개별 샘플로 저장 (embedding + label만)
        for window_idx in range(len(Zw)):
            sample_dict = {
                'embedding': torch.from_numpy(Zw[window_idx]),  # [256]
                'label': int(ds.Y[window_idx]) if ds.Y is not None else None  # 스칼라
            }
            # 파일명: {group}/{file}_{window_idx}.pt
            pt_path = group_out_dir / f'{file_stem}_{window_idx:04d}.pt'
            torch.save(sample_dict, pt_path)
        
        print(f"Saved: {group_name}/{file_stem} → {len(Zw)} samples")

        # 4-6) 이상 점수 계산
        if cfg.score_head == 'knn':
            # kNN: k개 최근접 이웃과의 평균 거리
            dists, _ = knn.kneighbors(Zw, n_neighbors=knn.n_neighbors, return_distance=True)
            scores = dists.mean(axis=1)
        else:
            # KDE: 로그 확률의 음수 (낮은 밀도 = 높은 이상 점수)
            logp = kde.score_samples(Zw)
            scores = -logp
        
        # 4-7) 이동 평균으로 스무딩
        scores = moving_average(scores, cfg.smooth)
        all_scores.append(scores)
        if Ys is not None:
            all_labels.append(ds.Y)

    # 5) 모든 파일의 결과 통합 (메트릭 계산용)
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels) if all_labels is not None else None

    # 6) 라벨이 있으면 메트릭 계산
    metrics = {}
    if labels is not None:
        pos = scores[labels==1]  # 이상 샘플들의 점수
        neg = scores[labels==0]  # 정상 샘플들의 점수
        
        # 메트릭 계산
        metrics['AUC_ROC'] = roc_auc_score(labels, scores)
        metrics['AUC_PR' ] = average_precision_score(labels, scores)
        
        # 최적 F1 점수 및 임계값 (argmax 사용)
        ps, rs, ths = precision_recall_curve(labels, scores)
        f1s = 2*ps*rs/(ps+rs+1e-8)
        best_idx = np.argmax(f1s)
        best_f1 = f1s[best_idx]
        best_thr = ths[best_idx] if best_idx < len(ths) else ths[-1]
        
        metrics['F1_best'] = float(best_f1)
        metrics['Threshold_F1_best'] = float(best_thr)
        
        # 최적 F1 임계값으로 예측 생성
        preds = (scores >= best_thr).astype(int)
        
        # F1-PA 스코어 (Point-Adjusted)
        metrics['F1_PA'] = compute_f1_pa(labels, preds)
        
        # KDE 교차점 임계값 (참고용)
        thr_kde = kde_crossing_threshold(pos, neg)
        metrics['Threshold_KDE_cross'] = float(thr_kde)
        
        print(json.dumps(metrics, indent=2))
        
        # 7) 결과 저장 (점수, 라벨, 예측)
        out_csv = Path(cfg.out_dir) / 'test_scores.csv'
        pd.DataFrame({'score':scores, 'label':labels, 'pred':preds}).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
    else:
        # 라벨이 없으면 점수만 저장
        out_csv = Path(cfg.out_dir) / 'test_scores_unlabeled.csv'
        pd.DataFrame({'score':scores}).to_csv(out_csv, index=False)
        print(f"Saved (unlabeled): {out_csv}")


# ---------------------------------------------------------
# 특성 추출: CSV/JSON 형식으로 전처리된 특성 저장
# ---------------------------------------------------------
def export_features(data_root: Path, out_dir: Path, splits=('train','test'), json_orient='records'):
    """모든 CSV 파일을 읽어 전처리된 특성을 CSV/JSON으로 저장
    
    read_skab_csv를 사용하여:
    - 라벨 및 시간 컬럼 제거
    - 숫자형 특성만 추출
    - CSV와 JSON 형식으로 저장
    
    저장 구조:
      {out_dir}/{split}/{파일명}_features.csv
      {out_dir}/{split}/{파일명}_features.json
      {out_dir}/{split}/feature_names.json  (특성 이름 리스트, split별 1회)
    
    Args:
        data_root: 데이터 루트 경로
        out_dir: 출력 디렉토리
        splits: 처리할 split 목록 (예: ('train', 'test'))
        json_orient: JSON 저장 형식
            - 'records': [{"f1":v1, "f2":v2}, ...] (행 중심)
            - 'split': {"columns":[...], "data":[[...], ...]} (테이블 형식)
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
            # 숫자형 특성만 추출 (라벨/시간 제거)
            df, y = read_skab_csv(p)
            stem = p.stem

            # CSV 저장
            csv_path = out_split / f"{stem}_features.csv"
            df.to_csv(csv_path, index=False)

            # JSON 저장
            json_path = out_split / f"{stem}_features.json"
            df.to_json(json_path, orient=json_orient)
            
            # 특성 이름은 split별로 한 번만 저장
            meta_path = out_split / "feature_names.json"
            if not meta_path.exists():
                with open(meta_path, "w") as f:
                    json.dump({"features": list(df.columns)}, f, ensure_ascii=False, indent=2)

            print(f"[export] {split}/{p.name} -> {csv_path.name}, {json_path.name}")



# ---------------------------------------------------------
# 명령줄 인터페이스 (CLI)
# ---------------------------------------------------------

def main():
    """메인 함수: 4개의 서브커맨드 제공
    
    1. train: 자기지도학습으로 TS2Vec 인코더 학습
    2. build_bank: 학습된 인코더로 메모리 뱅크 구축
    3. eval: 테스트 데이터 평가 및 메트릭 계산
    4. export_features: CSV/JSON 형식으로 특성 추출
    """
    parser = argparse.ArgumentParser(
        description='SKAB 시계열 이상 탐지 - TS2Vec 베이스라인'
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # 1) train: 자기지도학습
    p_train = sub.add_parser('train', help='자기지도학습으로 인코더 학습')
    p_train.add_argument('--data_root', type=Path, required=True, help='데이터 루트 경로')
    p_train.add_argument('--window', type=int, default=32, help='윈도우 길이')
    p_train.add_argument('--stride', type=int, default=1, help='윈도우 간격')
    p_train.add_argument('--batch', type=int, default=128, help='배치 크기')
    p_train.add_argument('--epochs', type=int, default=80, help='에포크 수')
    p_train.add_argument('--lr', type=float, default=1e-3, help='학습률')
    p_train.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    p_train.add_argument('--temp', type=float, default=0.1, help='contrastive loss temperature')
    p_train.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    p_train.add_argument('--out_dir', type=Path, default=Path('./outputs'), help='출력 디렉토리')

    # 2) build_bank: 메모리 뱅크 구축
    p_bank = sub.add_parser('build_bank', help='정상 데이터의 메모리 뱅크 구축')
    p_bank.add_argument('--data_root', type=Path, required=True, help='데이터 루트 경로')
    p_bank.add_argument('--window', type=int, default=32, help='윈도우 길이')
    p_bank.add_argument('--stride', type=int, default=1, help='윈도우 간격')
    p_bank.add_argument('--ckpt', type=Path, default=Path('./outputs/ts2vec_best.pt'), help='학습된 모델 경로')
    p_bank.add_argument('--out_dir', type=Path, default=Path('./outputs'), help='출력 디렉토리')
    p_bank.add_argument('--k_neighbors', type=int, default=10, help='kNN 이웃 개수')

    # 3) eval: 평가
    p_eval = sub.add_parser('eval', help='테스트 데이터 평가')
    p_eval.add_argument('--data_root', type=Path, required=True, help='데이터 루트 경로')
    p_eval.add_argument('--window', type=int, default=32, help='윈도우 길이')
    p_eval.add_argument('--stride', type=int, default=1, help='윈도우 간격')
    p_eval.add_argument('--ckpt', type=Path, default=Path('./outputs/ts2vec_best.pt'), help='학습된 모델 경로')
    p_eval.add_argument('--bank_dir', type=Path, default=Path('./outputs'), help='메모리 뱅크 디렉토리')
    p_eval.add_argument('--score_head', type=str, default='kde', choices=['kde','knn'], help='이상 점수 계산 방법')
    p_eval.add_argument('--smooth', type=int, default=5, help='스무딩 윈도우 크기')
    p_eval.add_argument('--out_dir', type=Path, default=Path('./outputs'), help='출력 디렉토리')

    # 4) export_features: 특성 추출
    p_dump = sub.add_parser('export_features', help='CSV/JSON으로 특성 추출')
    p_dump.add_argument('--data_root', type=Path, required=True, help='데이터 루트 경로')
    p_dump.add_argument('--out_dir', type=Path, default=Path('./feature_dumps'), help='출력 디렉토리')
    p_dump.add_argument('--splits', nargs='+', default=['train','test'], choices=['train','test'], help='처리할 split')
    p_dump.add_argument('--json_orient', type=str, default='records', choices=['records','split'], help='JSON 저장 형식')

    # 인자 파싱 및 실행
    args = parser.parse_args()

    # cmd를 미리 추출 (dataclass 생성 시 cmd 필드가 없으므로)
    cmd = getattr(args, "cmd", None)
    if hasattr(args, "cmd"):
        delattr(args, "cmd")

    # 서브커맨드별 실행
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
        raise ValueError(f"Unknown command: {cmd}")


if __name__ == '__main__':
    main()
