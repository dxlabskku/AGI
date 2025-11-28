#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_manifest_spotdiff.py
- spot-diff_pytorch/1cls/<object>/train|test|ground_truth 구조에 맞춰 manifest.csv 생성
- it_baseline_train.py 학습용 포맷 (id,img_paths,report_text,split)

사용 예시:
python make_manifest_spotdiff.py \
  --root /data/AGI/datasets/spot-diff_pytorch/1cls \
  --out /data/AGI/datasets/spot-diff_pytorch/manifest.csv \
  --val_from_train --val_ratio 0.1
"""
import os, csv, random, argparse
from pathlib import Path

def scan_split(obj_dir: Path, split_name: str):
    """각 object 폴더 안의 train/test 구조 순회"""
    rows = []
    if split_name == "train":
        base = obj_dir / "train" / "good"
        label_dirs = [("good", base)]
    else:
        base = obj_dir / "test"
        label_dirs = [("good", base / "good"), ("bad", base / "bad")]

    for label, root in label_dirs:
        if not root.exists():
            continue
        for img_path in sorted(root.rglob("*")):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            img_id = f"{obj_dir.name}_{img_path.stem}"
            caption = f"{obj_dir.name} {'normal sample' if label=='good' else 'defective sample'}"
            rows.append({
                "id": img_id,
                "img_paths": str(img_path.resolve()),
                "report_text": caption,
                "split": split_name,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="spot-diff_pytorch/1cls 디렉터리 경로")
    ap.add_argument("--out", required=True, help="출력 manifest.csv 경로")
    ap.add_argument("--val_from_train", action="store_true", help="train 데이터 일부를 val로 분할")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="val 비율")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    root = Path(args.root)
    assert root.exists(), f"경로가 존재하지 않습니다: {root}"

    rows_train, rows_val, rows_test = [], [], []

    # 각 object 순회
    for obj_dir in sorted(root.iterdir()):
        if not obj_dir.is_dir():
            continue
        # train
        train_rows = scan_split(obj_dir, "train")
        # test
        test_rows = scan_split(obj_dir, "test")

        # train 일부를 val로 이동
        if args.val_from_train and len(train_rows) > 2:
            n_val = max(1, int(len(train_rows) * args.val_ratio))
            random.shuffle(train_rows)
            val_rows = train_rows[:n_val]
            train_rows = train_rows[n_val:]
            for r in val_rows:
                r = dict(r)
                r["split"] = "val"
                rows_val.append(r)
        rows_train.extend(train_rows)
        rows_test.extend(test_rows)

    # CSV 저장
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","img_paths","report_text","split"])
        w.writeheader()
        for r in rows_train + rows_val + rows_test:
            w.writerow(r)

    print(f"[✓] manifest 생성 완료: {out_path}")
    print(f"  train: {len(rows_train)}, val: {len(rows_val)}, test: {len(rows_test)}")

if __name__ == "__main__":
    main()
