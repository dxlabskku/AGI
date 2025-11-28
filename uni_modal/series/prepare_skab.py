# 데이터셋 준비

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKAB downloader & preprocessing helper (minimal)
- Option A: Just preprocess local raw CSVs into train/test expected by series_ts2vec_baseline.py
- Option B: (Optional) Download a ZIP/TAR you already have locally and unpack to raw_dir

Usage
-----
# 1) (권장) 로컬에 받은 원본 CSV들을 정리만 할 때
python prepare_skab.py preprocess \
  --raw_dir /path/to/SKAB_raw \
  --out_dir /path/to/SKAB \
  --test_ratio 0.2

# 2) (선택) 로컬 ZIP을 먼저 풀고 정리할 때
python prepare_skab.py unpack_and_preprocess \
  --zip /path/to/SKAB.zip \
  --raw_dir /tmp/skab_raw \
  --out_dir /path/to/SKAB

Notes
-----
- 이 스크립트는 인터넷 다운로드를 수행하지 않습니다(로컬 ZIP/TAR만 처리).
- 라벨 컬럼 후보: ["anomaly", "label", "y", "class"]. 없으면 전체 정상(0)으로 간주합니다.
- 타임스탬프/인덱스 컬럼 후보: ["timestamp", "time", "datetime", "date", "index"]. 있으면 삭제합니다.
- 파일별로 이상비율>min_anom_ratio 이면 test로 분류, 아니면 무작위 분할(test_ratio)로 보냅니다.
- 결과는 다음 구조로 저장됩니다:
  out_dir/
    train/*.csv   # 정상 위주
    test/*.csv    # 혼합(이상 포함)
"""
from __future__ import annotations
import os, argparse, zipfile, tarfile, random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

LABEL_CANDIDATES = ["anomaly", "label", "y", "class"]
TIME_CANDIDATES  = ["timestamp", "time", "datetime", "date", "index"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_csvs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*.csv") if p.is_file()])


def load_csv_detect(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    y = None
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            y = df[c].astype(int)
            df = df.drop(columns=[c])
            break
    for c in TIME_CANDIDATES:
        if c in df.columns:
            df = df.drop(columns=[c])
    # 숫자형만 유지
    df = df.select_dtypes(include=[np.number])
    return df, y


def save_csv(dfX: pd.DataFrame, y: Optional[pd.Series], path: Path):
    if y is not None:
        out = dfX.copy()
        out["anomaly"] = y.values
    else:
        out = dfX
    out.to_csv(path, index=False)


def preprocess_cmd(raw_dir: Path, out_dir: Path, test_ratio: float, min_anom_ratio: float, seed: int):
    random.seed(seed)
    ensure_dir(out_dir / "train")
    ensure_dir(out_dir / "test")

    csvs = list_csvs(raw_dir)
    if not csvs:
        raise FileNotFoundError(f"No CSVs under {raw_dir}")

    for p in csvs:
        df = pd.read_csv(p)
        X, y = load_csv_detect(df)
        # 간단한 휴리스틱: 파일 내 이상 비율이 일정 이상이면 test로 보냄
        anom_ratio = float((y.mean() if y is not None else 0.0))
        target = "test" if (y is not None and anom_ratio >= min_anom_ratio) else ("test" if random.random() < test_ratio else "train")
        out_name = p.stem + ".csv"
        save_csv(X, y, out_dir / target / out_name)
    print(f"Preprocess done. Train: {len(list_csvs(out_dir/'train'))}, Test: {len(list_csvs(out_dir/'test'))}")


def unpack_and_preprocess(zip_path: Path, raw_dir: Path, out_dir: Path, test_ratio: float, min_anom_ratio: float, seed: int):
    ensure_dir(raw_dir)
    # 압축 해제 (zip/tar.gz 모두 지원)
    if zip_path.suffix == ".zip":
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(raw_dir)
    elif zip_path.suffixes[-2:] == ['.tar', '.gz'] or zip_path.suffix == '.tgz':
        with tarfile.open(zip_path, 'r:gz') as tf:
            tf.extractall(raw_dir)
    else:
        raise ValueError("Unsupported archive. Use .zip or .tar.gz")
    preprocess_cmd(raw_dir, out_dir, test_ratio, min_anom_ratio, seed)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('preprocess')
    p1.add_argument('--raw_dir', type=Path, required=True)
    p1.add_argument('--out_dir', type=Path, required=True)
    p1.add_argument('--test_ratio', type=float, default=0.2)
    p1.add_argument('--min_anom_ratio', type=float, default=0.01)  # 1% 이상이면 강제 test
    p1.add_argument('--seed', type=int, default=42)

    p2 = sub.add_parser('unpack_and_preprocess')
    p2.add_argument('--zip', type=Path, required=True, help='Local ZIP/TAR.GZ path')
    p2.add_argument('--raw_dir', type=Path, required=True)
    p2.add_argument('--out_dir', type=Path, required=True)
    p2.add_argument('--test_ratio', type=float, default=0.2)
    p2.add_argument('--min_anom_ratio', type=float, default=0.01)
    p2.add_argument('--seed', type=int, default=42)

    args = ap.parse_args()
    if args.cmd == 'preprocess':
        preprocess_cmd(args.raw_dir, args.out_dir, args.test_ratio, args.min_anom_ratio, args.seed)
    else:
        unpack_and_preprocess(args.zip, args.raw_dir, args.out_dir, args.test_ratio, args.min_anom_ratio, args.seed)

if __name__ == '__main__':
    main()
