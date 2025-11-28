#from train_utils import (build_model_from_sample, collate_mm, binary_metrics_from_logits,PTListDataset, NUM_CLASSES)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from train_utils_4multi import build_model_from_dataset, binary_metrics_from_logits, PTListDataset, NUM_CLASSES, collate_mm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os, json, argparse, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
import glob

def make_paths_json(root_dir: str, out_json: str, pattern: str="**/*.pt"):
    paths = sorted(glob.glob(os.path.join(root_dir, pattern), recursive=True))
    with open(out_json, "w") as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {len(paths)} paths to {out_json}")
    

def test(model, loader, device, criterion):
        EPS = 1e-12
        model.eval()
        val_loss_sum = 0.0; val_seen = 0
        v_correct = 0; vTP = vFP = vFN = 0
        with torch.no_grad():
            for feats, masks, y, _uids in tqdm(loader, total=len(loader)):
                feats = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
                masks = {k: v.to(device, non_blocking=True) for k, v in masks.items()}
                y     = y.to(device, non_blocking=True).long()

                logits = model(feats, masks)["logits"]
                loss   = criterion(logits, y)

                bs = y.size(0)
                val_loss_sum += loss.item() * bs
                val_seen     += bs

                preds = logits.argmax(dim=-1)
                v_correct += (preds == y).sum().item()
                vTP += ((preds == 1) & (y == 1)).sum().item()
                vFP += ((preds == 1) & (y == 0)).sum().item()
                vFN += ((preds == 0) & (y == 1)).sum().item()

        val_loss = val_loss_sum / max(1, val_seen)
        val_acc  = v_correct / max(1, val_seen)
        v_prec   = vTP / max(1, (vTP + vFP))
        v_rec    = vTP / max(1, (vTP + vFN))
        val_f1   = 2 * v_prec * v_rec / max(EPS, (v_prec + v_rec))

        print(f"[test] "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")
  
        
        
from tqdm import tqdm 

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.make_json:
       
        with open(args.total_json, "r") as f:
            allp = json.load(f)
        random.seed(0)
        random.shuffle(allp)
        n = len(allp)
        n_train = int(n * 0.9)
        with open(args.train_json, "w") as f:
            json.dump(allp[:n_train], f, indent=2, ensure_ascii=False)
        with open(args.val_json, "w") as f:
            json.dump(allp[n_train:], f, indent=2, ensure_ascii=False)
        print(f"[OK] Split: train={n_train}, val={n-n_train}")
    
        
    # --------------------
    # Dataset / Loader
    # --------------------
    train_ds = PTListDataset(args.train_json)
    val_ds   = PTListDataset(args.val_json)

  
    model = build_model_from_dataset(args.train_json, NUM_CLASSES).to(device)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_mm
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_mm
    )


    ce_loss  = torch.nn.CrossEntropyLoss()
   

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    save_dir = args.ckpt_dir
    os.makedirs(save_dir, exist_ok=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    EPS = 1e-12

    history = []
    for epoch in range(1, args.epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        run_loss = 0.0; n_seen = 0
        correct = 0; TP = FP = FN = 0

        for feats, masks, y, _uids in tqdm(train_dl, total=len(train_dl)):
            feats = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
            masks = {k: v.to(device, non_blocking=True) for k, v in masks.items()}
            y     = y.to(device, non_blocking=True).long()   # [B]

            logits = model(feats, masks)["logits"]           # [B,2]
            loss   = ce_loss(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bs = y.size(0)
            run_loss += loss.item() * bs
            n_seen   += bs

            preds = logits.argmax(dim=-1)                    # [B]
            correct += (preds == y).sum().item()
            TP += ((preds == 1) & (y == 1)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()

        train_loss = run_loss / max(1, n_seen)
        train_acc  = correct / max(1, n_seen)
        precision  = TP / max(1, (TP + FP))
        recall     = TP / max(1, (TP + FN))
        train_f1   = 2 * precision * recall / max(EPS, (precision + recall))

        # ---------- VAL ----------
        model.eval()
        val_loss_sum = 0.0; val_seen = 0
        v_correct = 0; vTP = vFP = vFN = 0
        with torch.no_grad():
            for feats, masks, y, _uids in tqdm(val_dl, total=len(val_dl)):
                feats = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
                masks = {k: v.to(device, non_blocking=True) for k, v in masks.items()}
                y     = y.to(device, non_blocking=True).long()

                logits = model(feats, masks)["logits"]
                loss   = ce_loss(logits, y)

                bs = y.size(0)
                val_loss_sum += loss.item() * bs
                val_seen     += bs

                preds = logits.argmax(dim=-1)
                v_correct += (preds == y).sum().item()
                vTP += ((preds == 1) & (y == 1)).sum().item()
                vFP += ((preds == 1) & (y == 0)).sum().item()
                vFN += ((preds == 0) & (y == 1)).sum().item()

        val_loss = val_loss_sum / max(1, val_seen)
        val_acc  = v_correct / max(1, val_seen)
        v_prec   = vTP / max(1, (vTP + vFP))
        v_rec    = vTP / max(1, (vTP + vFN))
        val_f1   = 2 * v_prec * v_rec / max(EPS, (v_prec + v_rec))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss,     "val_acc": val_acc,     "val_f1": val_f1
        })

        print(f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f}")
        # ---------- SAVE ----------

        ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "val_f1": val_f1,
            "val_acc": val_acc,
        }, ckpt_path)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/data/jupyter/AGI/datasets/spot-diff/output_features", help=".pt 파일 루트")
    p.add_argument("--pattern", type=str, default="**/*.pt")
    p.add_argument("--make_json", default=False, help="루트에서 PT 경로 스캔해 train/val JSON 생성")
    p.add_argument("--total_json", type=str, default="/data/jupyter/AGI/multiall_notime.json")
    p.add_argument("--train_json", type=str, default="/data/jupyter/AGI/encoders/uni_modal/series/outputs_final/train.json")
    p.add_argument("--val_json", type=str, default='/data/jupyter/AGI/encoders/uni_modal/series/outputs_final/test.json')
    p.add_argument("--test_json", type=str, default='/data/jupyter/AGI/encoders/uni_modal/series/outputs_final/test.json')
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--thr", type=float, default=0.5, help="멀티라벨 판정 threshold")
    p.add_argument("--ckpt_dir", type=str, default='/data/jupyter/AGI/series_ckpt', help='모델 가중치 저장 경로')
    p.add_argument("--ckpt_load", type=str, default='/data/jupyter/AGI/series_ckpt/epoch_034.pt', help='테스트 가중치 경로 ')          
    p.add_argument("--log_json", type=str, default="train_multi.json")
    p.add_argument("--train", default=True)
    args = p.parse_args() 
    if args.train:
        train(args)
    else:
        device='cuda'
        test_ds   = PTListDataset(args.test_json)
        test_dl = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_mm
        )
        criterion = torch.nn.CrossEntropyLoss()
        model = build_model_from_dataset(args.test_json, NUM_CLASSES).to(device)
        optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        checkpoint = torch.load(args.ckpt_load, map_location='cuda')
        model.load_state_dict(checkpoint["model_state"])
        test(model, test_dl, device, criterion)
    
