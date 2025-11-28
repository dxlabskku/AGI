import json
from pathlib import Path

files = [
    "./outputs/it_embed_train.jsonl",
    "./outputs/it_embed_val.jsonl",
    "./outputs/it_embed_test.jsonl",
]

merged = []
for f in files:
    split_name = Path(f).stem.replace("it_embed_", "")  # train/val/test
    with open(f, "r") as fin:
        for line in fin:
            item = json.loads(line)
            item["split"] = split_name  # ✅ split 정보 추가
            merged.append(item)

with open("./outputs/it_embed_all.jsonl", "w") as fout:
    for v in merged:
        fout.write(json.dumps(v, ensure_ascii=False) + "\n")

print(f"[✓] Merged {len(merged)} samples → it_embed_all.jsonl (split info added)")
