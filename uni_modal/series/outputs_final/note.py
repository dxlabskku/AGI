import os
import torch
from collections import Counter

def get_label_distribution(directory):
    label_counter = Counter()
    
    # 모든 pt 파일 순회
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            path = os.path.join(directory, filename)
            data = torch.load(path, map_location="cpu")
            
            # dict 구조에서 label 가져오기
            label = data.get("label", None)
            if label is not None:
                label_counter[label] += 1
    
    return label_counter


base_dir = "/data/jupyter/AGI/encoders/uni_modal/series/outputs_final"   # 여기에 train/val/test가 들어 있다고 가정

splits = ["train", "valid", "test"]

for split in splits:
    dir_path = os.path.join(base_dir, split)
    dist = get_label_distribution(dir_path)
    print(f"\n=== {split} label 분포 ===")
    for label, count in dist.items():
        print(f"label {label}: {count}개")
