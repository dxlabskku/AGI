
# Multimodal Anomaly Detection — Encoders Feature Format Guide

본 저장소는 멀티모달 이상탐지 모델에서 **인코더에서 추출된 임베딩(.pt)** 을 입력으로 사용합니다.  
즉, 어떤 인코더(오디오, 비디오, 시계열, 이미지텍스트 등)를 사용하더라도 **특정 포맷만 지키면 곧바로 본 모델을 학습**할 수 있습니다.

---

## 1. 입력으로 필요한 파일
###  PT 파일들 (`*.pt`)
각 샘플에 대해 임베딩 및 라벨이 저장된 `.pt` 파일이 필요합니다.
예시 ↓
<pre> 
  #비디오 모달
{
  "audio_tokens": Tensor (T,1,512) 또는 (T,512),
  "video_tokens": Tensor (T,7,8) 등,
  "labels": {"binary": 0 또는 1}  # 단일 또는 dict 가능
  "id": "..."   # (optional)
}
  #시계열 
{
  "embeddings": Tensor/ndarray [T, D],
  "labels": Tensor/ndarray/scalar 또는 None (0/1),
  "filename": "..."   # (optional)
}
  #이미지- 텍스트 페어 
{
  "img_feat": Tensor/ndarray [T_img, D_img] 또는 [D_img],
  "text_feat": Tensor/ndarray [T_txt, D_txt] 또는 [D_txt],
  "label": 0/1,
  "id" 또는 "filename": "..."  # (optional)
}
</pre>
###  PT 파일 경로를 모은 JSON
모든 `.pt` 파일의 경로 리스트가 저장된 `.json` 파일 1개  
예시 ↓
<pre> 
[
  "/data/jupyter/AGI/datasets/mimic-cxr-p10-mini/features/pt_p10316389/p10000032_s50414267.pt",
  "/data/jupyter/AGI/datasets/mimic-cxr-p10-mini/features/pt_p10316389/p10000032_s56699142.pt"
]
</pre>
### 2. 학습
<pre>
python main_multi.py \
  --root "/data/jupyter/AGI/datasets/spot-diff/output_features" \
  --pattern "**/*.pt" \
  --make_json True \
  --total_json "/data/jupyter/AGI/multiall.json" \
  --batch_size 8 \
  --lr 3e-4 \
  --epochs 50 \
  --thr 0.5 \
  --ckpt_dir "/data/jupyter/AGI/multi_ckpt" \
  --log_json "train_multi.json" \
  --train True
</pre>
### 3. 평가
<pre>
python main_multi.py \
  --test_json "/data/jupyter/AGI/multiall_test.json" \
  --ckpt_load "/data/jupyter/AGI/series_ckpt/epoch_034.pt" \
  --train False
</pre>
### 4. 디렉토리
<pre> 
AGI/
 ├── encoders/
 │    ├── uni_modal/
 │    ├── multi_modal/
 │    └── ...
 ├── main_multi.py
 ├── README.md
 └── requirements.txt
  </pre>
