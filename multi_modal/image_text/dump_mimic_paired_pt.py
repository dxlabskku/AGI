# 레이블 파일 (mimic-cxr-2.0.0-chexpert.csv.gz) 반영하는걸로 수정 

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import pydicom
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoTokenizer

CHEXPERT_14 = [
    "No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices"
]

def dcm_to_pil(dcm_path: str) -> Image.Image:
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    arr = arr - arr.min()
    denom = (arr.max() - arr.min()) + 1e-6
    arr = (arr / denom) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    return img

@torch.no_grad()
def encode_image(clip_model, clip_proc, imgs, device):
    feats = []
    for img in imgs:
        inputs = clip_proc(images=img, return_tensors="pt").to(device)
        feat = clip_model.get_image_features(**inputs)
        feat = F.normalize(feat, dim=-1)
        feats.append(feat.squeeze(0))
    return torch.stack(feats).mean(0)

@torch.no_grad()
def encode_text(e5_model, e5_tok, text, device):
    prompt = f"passage: {text.strip() or 'empty report'}"
    inputs = e5_tok([prompt], return_tensors="pt", truncation=True, max_length=512).to(device)
    feat = e5_model(**inputs).last_hidden_state[:, 0]
    feat = F.normalize(feat, dim=-1)
    return feat.squeeze(0)

def collect_studies(root_p10: Path, stop_pat: str):
    patients = sorted([p for p in root_p10.iterdir() if p.is_dir() and p.name.startswith("p")])
    studies = []
    for p in patients:
        if p.name > stop_pat:
            break
        for s in sorted(p.glob("s*")):
            txt = p / f"{s.name}.txt"
            dcm = sorted(s.glob("*.dcm"))
            if not dcm:
                continue
            studies.append({
                "patient_id": p.name,
                "study_id": s.name,            # e.g. "s50414267"
                "study_id_int": int(s.name[1:]),
                "dcm_paths": [str(x) for x in dcm],
                "txt_path": str(txt) if txt.exists() else None
            })
    return studies

def load_chexpert_map(csv_path: str, uncertain_as_positive: bool):
    df = pd.read_csv(csv_path, compression="infer")
    cols = ["study_id"] + [c for c in CHEXPERT_14 if c in df.columns]
    df = df[cols].drop_duplicates("study_id")
    df.set_index("study_id", inplace=True)

    def row_to_dict(row):
        labels_14 = {}
        for k in CHEXPERT_14:
            if k not in row.index:
                continue
            v = row[k]
            if pd.isna(v):
                labels_14[k] = None
            else:
                labels_14[k] = float(v)
        vals = []
        for k,v in labels_14.items():
            if k == "No Finding": 
                continue
            if v is None:
                continue
            if v == 1.0 or (uncertain_as_positive and v == -1.0):
                vals.append(1)
        label_bin = 1 if len(vals) > 0 else 0
        return label_bin, labels_14

    chex_map = {}
    for sid, row in df.iterrows():
        chex_map[int(sid)] = row_to_dict(row)
    return chex_map

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_p10", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--chexpert_csv", required=True, help="mimic-cxr-2.0.0-chexpert.csv(.gz)")
    parser.add_argument("--stop_at", default="p10075034")
    parser.add_argument("--uncertain_as_positive", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    out_dir = Path(args.out_dir)
    pt_dir = out_dir / "pt"
    pt_dir.mkdir(parents=True, exist_ok=True)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    e5_model = AutoModel.from_pretrained("intfloat/e5-base").to(device).eval()
    e5_tok = AutoTokenizer.from_pretrained("intfloat/e5-base")

    studies = collect_studies(Path(args.root_p10), args.stop_at)
    chex_map = load_chexpert_map(args.chexpert_csv, args.uncertain_as_positive)
    print(f"[info] collected {len(studies)} studies (<= {args.stop_at})")
    print(f"[info] chexpert map size: {len(chex_map)}")

    miss, pos, neg = 0, 0, 0
    for i, item in enumerate(studies, 1):
        imgs = [dcm_to_pil(p) for p in item["dcm_paths"]]
        img_feat = encode_image(clip_model, clip_proc, imgs, device)

        report_text = ""
        if item["txt_path"] and os.path.exists(item["txt_path"]):
            with open(item["txt_path"], "r", encoding="utf-8", errors="ignore") as f:
                report_text = f.read()
        text_feat = encode_text(e5_model, e5_tok, report_text, device)

        if item["study_id_int"] in chex_map:
            label_bin, labels_14 = chex_map[item["study_id_int"]]
        else:
            label_bin, labels_14 = 0, {k: None for k in CHEXPERT_14}
            miss += 1

        save_path = pt_dir / f"{item['patient_id']}_{item['study_id']}.pt"
        torch.save({
            "patient_id": item["patient_id"],
            "study_id": item["study_id"],
            "img_feat": img_feat.cpu(),
            "text_feat": text_feat.cpu(),
            "label_bin": label_bin,
            "labels_14": labels_14,
            "label_source": "chexpert",
        }, save_path)

        pos += (label_bin == 1)
        neg += (label_bin == 0)
        if i % 10 == 0:
            print(f"[{i}/{len(studies)}] saved {save_path.name}  bin={label_bin}")

    print(f"[stats] +:{pos}  -:{neg}  miss(join):{miss}")
    print(f"[done] saved all .pt files to {pt_dir}")

if __name__ == "__main__":
    main()


# import os, json, glob
# from pathlib import Path
# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# import pydicom
# from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoTokenizer

# def dcm_to_pil(dcm_path: str) -> Image.Image:
#     """DICOM 파일을 PIL 이미지로 변환"""
#     ds = pydicom.dcmread(dcm_path)
#     arr = ds.pixel_array.astype(np.float32)
#     arr = arr - arr.min()
#     denom = (arr.max() - arr.min()) + 1e-6
#     arr = (arr / denom) * 255.0
#     arr = np.clip(arr, 0, 255).astype(np.uint8)
#     img = Image.fromarray(arr).convert("RGB")
#     return img

# def default_labeler(report_text: str):
#     """간단한 규칙 기반 라벨"""
#     if "pulmonary embol" in report_text.lower():
#         return 1
#     return 0

# @torch.no_grad()
# def encode_image(clip_model, clip_proc, imgs, device):
#     """여러 장의 DICOM → 평균 임베딩"""
#     feats = []
#     for img in imgs:
#         inputs = clip_proc(images=img, return_tensors="pt").to(device)
#         feat = clip_model.get_image_features(**inputs)
#         feat = F.normalize(feat, dim=-1)
#         feats.append(feat.squeeze(0))
#     return torch.stack(feats).mean(0)

# @torch.no_grad()
# def encode_text(e5_model, e5_tok, text, device):
#     """텍스트 인코딩"""
#     prompt = f"passage: {text.strip() or 'empty report'}"
#     inputs = e5_tok([prompt], return_tensors="pt", truncation=True, max_length=512).to(device)
#     feat = e5_model(**inputs).last_hidden_state[:, 0]
#     feat = F.normalize(feat, dim=-1)
#     return feat.squeeze(0)

# def collect_studies(root_p10: Path, stop_pat: str):
#     """p10/<patient>/<study> 구조 순회"""
#     patients = sorted([p for p in root_p10.iterdir() if p.is_dir() and p.name.startswith("p")])
#     studies = []
#     for p in patients:
#         if p.name > stop_pat:
#             break
#         for s in sorted(p.glob("s*")):
#             txt = p / f"{s.name}.txt"
#             dcm = sorted(s.glob("*.dcm"))
#             if not dcm:
#                 continue
#             studies.append({
#                 "patient_id": p.name,
#                 "study_id": s.name,
#                 "dcm_paths": [str(x) for x in dcm],
#                 "txt_path": str(txt) if txt.exists() else None
#             })
#     return studies

# def main():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--root_p10", required=True)
#     parser.add_argument("--out_dir", required=True)
#     parser.add_argument("--stop_at", default="p10015725")
#     parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     args = parser.parse_args()

#     device = args.device
#     out_dir = Path(args.out_dir)
#     pt_dir = out_dir / "pt"
#     pt_dir.mkdir(parents=True, exist_ok=True)

#     # 모델 로드 (freeze)
#     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
#     clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     e5_model = AutoModel.from_pretrained("intfloat/e5-base").to(device).eval()
#     e5_tok = AutoTokenizer.from_pretrained("intfloat/e5-base")

#     studies = collect_studies(Path(args.root_p10), args.stop_at)
#     print(f"[info] collected {len(studies)} studies (<= {args.stop_at})")

#     for i, item in enumerate(studies, 1):
#         imgs = [dcm_to_pil(p) for p in item["dcm_paths"]]
#         img_feat = encode_image(clip_model, clip_proc, imgs, device)

#         report_text = ""
#         if item["txt_path"] and os.path.exists(item["txt_path"]):
#             with open(item["txt_path"], "r", encoding="utf-8", errors="ignore") as f:
#                 report_text = f.read()
#         text_feat = encode_text(e5_model, e5_tok, report_text, device)
#         label = default_labeler(report_text)

#         save_path = pt_dir / f"{item['patient_id']}_{item['study_id']}.pt"
#         torch.save({
#             "patient_id": item["patient_id"],
#             "study_id": item["study_id"],
#             "img_feat": img_feat.cpu(),
#             "text_feat": text_feat.cpu(),
#             "label": label
#         }, save_path)

#         if i % 10 == 0:
#             print(f"[{i}/{len(studies)}] saved {save_path.name}")

#     print(f"[done] saved all .pt files to {pt_dir}")

# if __name__ == "__main__":
#     main()
