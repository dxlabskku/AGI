#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import argparse
import random
from pathlib import Path
from typing import List, Dict

import pandas as pd


def find_studies(base: Path, subsets: List[str], require_dcm: bool, min_images: int) -> List[Dict]:
    rows = []
    files_dir = base / 'files'
    for subset in subsets:
        root = files_dir / subset
        if not root.exists():
            print(f"[warn] subset not found: {root}")
            continue
        # patient dirs: p10*/
        for pdir in sorted(root.glob('p*/')):
            subject = pdir.name
            # study dirs: sXXXXXXX/
            for sdir in sorted(pdir.glob('s*/')):
                study_id = sdir.name
                dcms = sorted([str(x.resolve()) for x in sdir.glob('*.dcm')])
                if require_dcm and len(dcms) < min_images:
                    continue
                rep_path = pdir / f"{study_id}.txt"
                report_text = ""
                if rep_path.exists():
                    try:
                        report_text = rep_path.read_text(encoding='utf-8', errors='ignore').strip()
                    except Exception:
                        pass
                rows.append({
                    'id': study_id,
                    'img_paths': "|".join(dcms),
                    'report_text': report_text,
                    'subject': subject,
                    'view': "",
                    'time': "",
                })
    return rows


def split_by_subject(rows: List[Dict], train_ratio: float, val_ratio: float, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    by_subj: Dict[str, List[Dict]] = {}
    for r in rows:
        by_subj.setdefault(r['subject'], []).append(r)
    subs = list(by_subj.keys())
    random.shuffle(subs)
    n = len(subs)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    train_subs = set(subs[:n_tr])
    val_subs   = set(subs[n_tr:n_tr+n_va])
    test_subs  = set(subs[n_tr+n_va:])

    out = []
    for s, grp in by_subj.items():
        split = 'train' if s in train_subs else ('val' if s in val_subs else 'test')
        for r in grp:
            rr = dict(r)
            rr['split'] = split
            out.append(rr)
    return out


def main():
    ap = argparse.ArgumentParser(description='Make MIMIC-CXR manifest CSV')
    ap.add_argument('--base', type=str, required=True,
                    help='Base dir to MIMIC-CXR v2.1.0 (contains files/, cxr-*.csv.gz, etc.)')
    ap.add_argument('--subsets', type=str, default='p10',
                    help='Comma-separated subset names under files/ (e.g., p10 or p10,p11)')
    ap.add_argument('--out', type=str, required=True,
                    help='Output CSV path')
    ap.add_argument('--require_dcm', action='store_true', default=True,
                    help='Keep only rows that have at least min_images DICOMs (default: True)')
    ap.add_argument('--min_images', type=int, default=1,
                    help='Minimum number of DICOMs required per study (default: 1)')
    ap.add_argument('--train_ratio', type=float, default=0.8)
    ap.add_argument('--val_ratio',   type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)

    args = ap.parse_args()
    base = Path(args.base)
    subsets = [s.strip() for s in args.subsets.split(',') if s.strip()]

    print(f"[info] scanning base={base} subsets={subsets}")
    rows = find_studies(base, subsets, require_dcm=args.require_dcm, min_images=args.min_images)
    print(f"[info] collected studies: {len(rows)}")

    # remove empty rows if require_dcm=False but no images and no text
    rows = [r for r in rows if r.get('img_paths', '') or r.get('report_text', '')]

    # split by subject
    rows = split_by_subject(rows, args.train_ratio, args.val_ratio, seed=args.seed)

    # save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=['id','img_paths','report_text','split','subject','view','time'])
    df.to_csv(out_path, index=False)
    print(f"[done] wrote manifest: {out_path} (rows={len(df)})")


if __name__ == '__main__':
    main()
