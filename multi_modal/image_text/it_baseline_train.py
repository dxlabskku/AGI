#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, io, json, math, random, argparse, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from PIL import Image
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

from transformers import (
    AutoTokenizer, AutoModel,
    CLIPTextModel, CLIPVisionModel,
    CLIPImageProcessor, CLIPTokenizer,
)
from tqdm import tqdm

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def exists(x):
    return x is not None

# -----------------------------
# Config
# -----------------------------

@dataclass
class CFG:
    manifest: str = "./mimic_manifest.csv"
    out_dir: str = "./outputs"
    ckpt: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # model dims
    dim: int = 256
    dropout: float = 0.1
    learnable_tau: bool = True
    tau_init: float = 0.07

    # training
    epochs: int = 5
    batch_size: int = 64
    lr_p: float = 1e-4     # projector
    lr_f: float = 5e-5     # fuser
    wd: float = 1e-2
    warmup_steps: int = 200
    grad_accum: int = 1
    fp16: bool = True

    # loss type: 'single' uses fused z_txt only / 'multi' treats clip & e5 as multi-positive
    loss: str = "single"  # or 'multi'

    # dataloader
    num_workers: int = 4
    max_txt_len_e5: int = 192

    # dump
    dump_only: bool = False
    dump_split: str = "val"  # which split to dump

# -----------------------------
# Data
# -----------------------------

class ITManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_processor: CLIPImageProcessor, max_txt_len_e5: int = 192):
        self.df = df.reset_index(drop=True)
        self.image_processor = image_processor
        self.max_txt_len_e5 = max_txt_len_e5
        self.img_tf = T.Compose([
            T.Resize(384, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
        ])

    def __len__(self):
        return len(self.df)

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
        except Exception:
            # 이미지 로드 실패 시 흰색 이미지로 대체
            im = Image.new("RGB", (224, 224), color=(255, 255, 255))
        im = self.img_tf(im)
        pixel = self.image_processor(images=im, return_tensors="pt")['pixel_values'][0]  # (3,224,224)
        return pixel

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sid = str(row.get('id', idx))
        raw_paths = str(row['img_paths']) if exists(row.get('img_paths')) else ''
        img_paths = [p for p in raw_paths.split('|') if p]
        text = str(row.get('report_text', '') or '')

        # 멀티뷰 평균 (간단)
        imgs = []
        for p in img_paths[:4]:  # 최대 4장까지
            imgs.append(self._load_image_tensor(p))
        if len(imgs) == 0:
            # 결측 처리: dummy image
            imgs = [self._load_image_tensor("")]
        images = torch.stack(imgs, dim=0)  # (V,3,224,224)

        meta = {
            'id': sid,
            'subject': str(row.get('subject', '')),
            'view': str(row.get('view', '')),
            'time': str(row.get('time', '')),
            'split': str(row.get('split', '')),
        }
        item = {
            'images': images,         # (V,3,224,224)
            'text': text,             # raw string
            'meta': meta,
        }
        return item


def collate_fn(batch: List[Dict]):
    # 가변 길이 멀티뷰를 평균 풀링하여 1장으로 축소
    imgs = []
    texts_clip = []
    texts_e5 = []
    masks_img = []
    masks_txt = []
    metas = []

    for b in batch:
        v_imgs = b['images']  # (V,3,224,224)
        img = v_imgs.mean(dim=0)  # (3,224,224)
        imgs.append(img)
        txt = b['text']
        texts_clip.append(txt)
        texts_e5.append(txt)
        masks_img.append(1 if v_imgs is not None else 0)
        masks_txt.append(1 if len(txt.strip()) > 0 else 0)
        metas.append(b['meta'])

    images = torch.stack(imgs, dim=0)  # (B,3,224,224)
    mask = torch.tensor(np.stack([masks_img, masks_txt], axis=1)).bool()  # (B,2)

    return {
        'images': images,
        'texts': {'clip': texts_clip, 'e5': texts_e5},
        'mask': mask,
        'meta': metas,
    }

# -----------------------------
# Model
# -----------------------------

class ProjectionHead(nn.Module):
    def __init__(self, dim_in: int, dim_out: int = 256, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 512),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(512, dim_out),
            nn.LayerNorm(dim_out),
        )
    def forward(self, x):
        return l2_normalize(self.net(x))


class GatedFuser(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1)
        )
    def forward(self, z_clip, z_e5):
        g = torch.sigmoid(self.gate(torch.cat([z_clip, z_e5], dim=-1)))  # (B,1)
        return g * z_clip + (1 - g) * z_e5, g.squeeze(-1)


class ITBackbone(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        # Vision
        self.vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        v_dim = self.vision.config.hidden_size  # 768
        # Text: CLIP
        self.txt_clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        t_dim_clip = self.txt_clip.config.hidden_size  # 512
        # Text: e5-base
        self.txt_e5 = AutoModel.from_pretrained("intfloat/e5-base")
        t_dim_e5 = self.txt_e5.config.hidden_size  # 768

        # Freeze encoders (baseline)
        for m in [self.vision, self.txt_clip, self.txt_e5]:
            for p in m.parameters():
                p.requires_grad = False

        # Projectors
        self.p_img = ProjectionHead(v_dim, cfg.dim, cfg.dropout)
        self.p_clip = ProjectionHead(t_dim_clip, cfg.dim, cfg.dropout)
        self.p_e5 = ProjectionHead(t_dim_e5, cfg.dim, cfg.dropout)
        # Fuser
        self.fuser = GatedFuser(cfg.dim)

        # Temperature
        tau = torch.tensor([cfg.tau_init], dtype=torch.float32)
        self._log_tau = nn.Parameter(torch.log(tau)) if cfg.learnable_tau else nn.Parameter(torch.log(tau), requires_grad=False)

        # Tokenizers / processors
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.e5_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")

    def tau(self):
        return torch.clamp(self._log_tau.exp(), 1e-3, 1.0)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B,3,224,224) in CLIP space already
        out = self.vision(pixel_values=images)
        h = out.pooler_output  # (B, hidden)
        return self.p_img(h)

    def encode_text_clip(self, texts: List[str]) -> torch.Tensor:
        tok = self.clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        tok = {k: v.to(self._log_tau.device) for k, v in tok.items()}
        out = self.txt_clip(**tok)
        h = out.pooler_output  # (B, hidden)
        return self.p_clip(h)

    def encode_text_e5(self, texts: List[str]) -> torch.Tensor:
        # e5: mean-pool last hidden states (CLS-pooled also possible)
        tok = self.e5_tokenizer(texts, padding=True, truncation=True, max_length=192, return_tensors="pt")
        tok = {k: v.to(self._log_tau.device) for k, v in tok.items()}
        out = self.txt_e5(**tok)
        last = out.last_hidden_state  # (B,T,H)
        attn = tok['attention_mask'].unsqueeze(-1)  # (B,T,1)
        summed = (last * attn).sum(dim=1)
        denom = attn.sum(dim=1).clamp(min=1e-6)
        h = summed / denom  # mean-pool
        return self.p_e5(h)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        images = batch['images'].to(self._log_tau.device)
        z_img = self.encode_image(images)
        z_c = self.encode_text_clip(batch['texts']['clip'])
        z_e = self.encode_text_e5(batch['texts']['e5'])
        z_txt, gate = self.fuser(z_c, z_e)
        # presence & quality (간단 예: 텍스트 길이 기반)
        B = images.size(0)
        presence = batch['mask'].to(self._log_tau.device)  # (B,2)
        txt_len = torch.tensor([len(t) for t in batch['texts']['clip']], device=self._log_tau.device, dtype=torch.float32)
        q_img = torch.ones(B, device=self._log_tau.device)  # placeholder 1.0
        q_txt = torch.tanh((txt_len / 256.0))  # 0~1 approx
        quality = torch.stack([q_img, q_txt], dim=-1)  # (B,2)

        # align score (for logging)
        s_align = 1.0 - (z_img * z_txt).sum(dim=-1)  # cosine since both are L2-normed

        return {
            'z_img': z_img, 'z_txt': z_txt,
            'z_txt_clip': z_c, 'z_txt_e5': z_e,
            'gate': gate, 'presence': presence, 'quality': quality,
            's_align': s_align,
        }

# -----------------------------
# Losses & Metrics
# -----------------------------

def info_nce_single(z_img: torch.Tensor, z_txt: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    # standard contrastive (image <-> fused text)
    # sim = z @ z^T because L2 normalized → cosine
    sim_i2t = (z_img @ z_txt.t()) / tau
    sim_t2i = sim_i2t.t()
    labels = torch.arange(z_img.size(0), device=z_img.device)
    loss_i = F.cross_entropy(sim_i2t, labels)
    loss_t = F.cross_entropy(sim_t2i, labels)
    return 0.5 * (loss_i + loss_t)


def info_nce_multi_pos(z_img: torch.Tensor, z_c: torch.Tensor, z_e: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    # multi-positive: combine CLIP-text & e5-text as positives
    S_ic = (z_img @ z_c.t()) / tau
    S_ie = (z_img @ z_e.t()) / tau
    # log-sum-exp of positives on the diagonal
    B = z_img.size(0)
    diag_ic = torch.diag(S_ic)
    diag_ie = torch.diag(S_ie)
    pos = torch.stack([diag_ic, diag_ie], dim=0)  # (2,B)
    pos = torch.logsumexp(pos, dim=0)  # (B,)

    # denominator: all texts except we need to avoid double-counting positives
    denom = torch.logsumexp(torch.cat([S_ic, S_ie], dim=1), dim=1)  # (B,)
    loss_i = -(pos - denom).mean()

    # symmetric term (texts -> image)
    S_ci = S_ic.t(); S_ei = S_ie.t()
    diag_ci = torch.diag(S_ci)
    diag_ei = torch.diag(S_ei)
    pos_t = torch.stack([diag_ci, diag_ei], dim=0)
    pos_t = torch.logsumexp(pos_t, dim=0)
    denom_t = torch.logsumexp(torch.cat([S_ci, S_ei], dim=1), dim=1)
    loss_t = -(pos_t - denom_t).mean()

    return 0.5 * (loss_i + loss_t)

# -----------------------------
# Training / Evaluation / Dump
# -----------------------------

def make_loaders(cfg: CFG, manifest_path: str, image_processor) -> Tuple[DataLoader, DataLoader, DataLoader]:
    df = pd.read_csv(manifest_path)
    assert 'split' in df.columns, "manifest에 split(train/val/test) 컬럼이 필요합니다."
    dsets = {}
    for sp in ['train', 'val', 'test']:
        sub = df[df['split'] == sp].copy()
        dsets[sp] = ITManifestDataset(sub, image_processor=image_processor, max_txt_len_e5=cfg.max_txt_len_e5)
    loaders = {
        sp: DataLoader(dsets[sp], batch_size=cfg.batch_size, shuffle=(sp=='train'),
                       num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn)
        for sp in dsets
    }
    return loaders['train'], loaders['val'], loaders['test']


def save_ckpt(path: str, model: ITBackbone, optim: torch.optim.Optimizer, step: int, cfg: CFG):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'step': step,
        'cfg': cfg.__dict__,
    }, path)


def load_ckpt(path: str, model: ITBackbone, optim: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=True)
    if optim is not None and 'optim' in ckpt:
        try:
            optim.load_state_dict(ckpt['optim'])
        except Exception:
            pass
    return ckpt.get('step', 0)


def train(cfg: CFG):
    set_seed(cfg.seed)
    device = cfg.device

    # model
    model = ITBackbone(cfg).to(device)

    # data
    train_loader, val_loader, _ = make_loaders(cfg, cfg.manifest, model.image_processor)

    # params: projectors + fuser + (optional) tau
    params = list(model.p_img.parameters()) + list(model.p_clip.parameters()) + list(model.p_e5.parameters()) + list(model.fuser.parameters())
    if model._log_tau.requires_grad:
        params += [model._log_tau]

    optim = torch.optim.AdamW(params, lr=cfg.lr_p, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16 and device.startswith('cuda'))

    global_step = 0

    model.train()

    best_val = float("inf")   # s_align는 낮을수록 좋음
    best_epoch = -1
    no_improve = 0

    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        running_loss = 0.0
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            with torch.cuda.amp.autocast(enabled=cfg.fp16 and device.startswith('cuda')):
                out = model(batch)
                tau = model.tau()
                if cfg.loss == 'multi':
                    loss = info_nce_multi_pos(out['z_img'], out['z_txt_clip'], out['z_txt_e5'], tau)
                else:
                    loss = info_nce_single(out['z_img'], out['z_txt'], tau)

            scaler.scale(loss / cfg.grad_accum).backward()
            if step % cfg.grad_accum == 0:
                scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
            running_loss += loss.item()

            if step % 20 == 0:
                pbar.set_postfix({
                    'loss': f"{running_loss/step:.4f}",
                    'tau': f"{float(tau):.4f}",
                })
            global_step += 1

        # 간단한 validation: 평균 s_align
        val_align = evaluate_align(cfg, model, val_loader)
        print(f"[Val] mean s_align(fused): {val_align:.4f}")

        # val_align = 현재 에폭에서 계산된 mean s_align(fused)
        if cfg.save_best:
            if val_align < best_val:
                best_val = val_align
                best_epoch = epoch
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "best_val": best_val,
                }, os.path.join(cfg.out_dir, "ckpt_best.pt"))
                print(f"[Best] epoch={epoch}  val_s_align={best_val:.4f}  -> saved ckpt_best.pt")
            else:
                no_improve += 1
                if cfg.patience > 0 and no_improve >= cfg.patience:
                    print(f"[EarlyStop] no improvement for {cfg.patience} epochs since best@{best_epoch}. Stop.")
                    break


        # save
        ckpt_path = os.path.join(cfg.out_dir, 'ckpt.pt')
        save_ckpt(ckpt_path, model, optim, global_step, cfg)
        print(f"[Save] {ckpt_path}")


@torch.no_grad()
def evaluate_align(cfg: CFG, model: ITBackbone, loader: DataLoader) -> float:
    device = cfg.device
    model.eval()
    vals = []
    for batch in loader:
        out = model(batch)
        vals.append(out['s_align'].detach().cpu().numpy())
    model.train()
    if len(vals) == 0:
        return 0.0
    return float(np.concatenate(vals).mean())


@torch.no_grad()
# def dump_embeddings(cfg: CFG):
def dump_embeddings(cfg, split=None):
    if split is not None:
        cfg.dump_split = split
    print(f"[Dump] split={cfg.dump_split}")
    
    device = cfg.device
    model = ITBackbone(cfg).to(device)
    assert cfg.ckpt is not None and os.path.exists(cfg.ckpt), "--ckpt 경로가 필요합니다."
    load_ckpt(cfg.ckpt, model)
    model.eval()

    _, val_loader, test_loader = make_loaders(cfg, cfg.manifest, model.image_processor)
    loader = val_loader if cfg.dump_split == 'val' else (test_loader if cfg.dump_split == 'test' else val_loader)

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, f"it_embed_{cfg.dump_split}.jsonl")
    with open(out_path, 'w', encoding='utf-8') as f:
        for batch in tqdm(loader, desc=f"Dump {cfg.dump_split}"):
            out = model(batch)
            z_img = out['z_img'].detach().cpu().numpy()
            z_txt = out['z_txt'].detach().cpu().numpy()
            z_c = out['z_txt_clip'].detach().cpu().numpy()
            z_e = out['z_txt_e5'].detach().cpu().numpy()
            s_align = out['s_align'].detach().cpu().numpy()
            pres = out['presence'].detach().cpu().numpy()
            qual = out['quality'].detach().cpu().numpy()

            for i, meta in enumerate(batch['meta']):
                rec = {
                    'id': meta.get('id', str(i)),
                    'z_img': z_img[i].tolist(),
                    'z_txt': z_txt[i].tolist(),
                    'z_txt_clip': z_c[i].tolist(),
                    'z_txt_e5': z_e[i].tolist(),
                    'mask': pres[i].astype(int).tolist(),
                    'quality': [float(qual[i,0]), float(qual[i,1])],
                    'meta': meta,
                    's_align': float(s_align[i]),
                    'version': 'it-backbone-clip-e5-v1',
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[Dump] {out_path}")

# -----------------------------
# Main
# -----------------------------

def parse_args() -> CFG:
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=str, default=CFG.manifest)
    p.add_argument('--out_dir', type=str, default=CFG.out_dir)
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--epochs', type=int, default=CFG.epochs)
    p.add_argument('--batch_size', type=int, default=CFG.batch_size)
    p.add_argument('--dim', type=int, default=CFG.dim)
    p.add_argument('--loss', type=str, default=CFG.loss, choices=['single','multi'])
    p.add_argument('--dump_only', action='store_true')
    # p.add_argument('--dump_split', type=str, default=CFG.dump_split, choices=['train','val','test'])
    p.add_argument('--dump_split', type=str, default=CFG.dump_split, choices=['train','val','test','all'])
    p.add_argument('--seed', type=int, default=CFG.seed)
    p.add_argument("--save_best", action="store_true", help="save best checkpoint by val s_align")
    p.add_argument("--patience", type=int, default=0, help="early stop patience (0=off)")

    args = p.parse_args()
    cfg = CFG()
    cfg.manifest = args.manifest
    cfg.out_dir = args.out_dir
    cfg.ckpt = args.ckpt
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.dim = args.dim
    cfg.loss = args.loss
    cfg.dump_only = args.dump_only
    cfg.dump_split = args.dump_split
    cfg.seed = args.seed
    cfg.save_best = args.save_best
    cfg.patience  = args.patience

    return cfg


if __name__ == '__main__':
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    # if cfg.dump_only:
    #     dump_embeddings(cfg)
    
    if cfg.dump_only:
        assert cfg.ckpt is not None, "dump_only 모드에서는 --ckpt가 필요합니다."
        splits = ['train','val','test'] if cfg.dump_split == 'all' else [cfg.dump_split]
        for sp in splits:
            dump_embeddings(cfg, split=sp)  # 여기서 실제 임베딩 추출
        sys.exit(0)
    else:
        train(cfg)  # 평소처럼 학습 수행
