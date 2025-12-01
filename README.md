
# Multimodal Anomaly Detection — Encoders Feature Format Guide

본 멀티모달 이상탐지 모델은 **인코더에서 추출된 임베딩(.pt)** 을 입력으로 사용합니다.  
인코더의 종류는 자유로우며, 임베딩 차원을 256이상으로 설정하는 것을 권장합니다. 

본 저장소는 텍스트·이미지·비디오·오디오·시계열 등 서로 다른 형태의 데이터를 단일 파이프라인에서 처리할 수 있는 멀티모달 이상 탐지 모델을 구현한다. 사전학습된 인코더들을 고정 특징 추출기로 활용하여 모달리티별 표현을 확보하고, 이를 공통 차원으로 정규화한 뒤 시점·모달리티 임베딩을 결합하여 통합 토큰 시퀀스를 구성한다. 이후 Perceiver 기반 Latent Attention 모듈을 통해 입력 길이와 모달 구성 변화에 견고한 잠재 표현을 학습하며, Mixture-of-Experts(MoE) 융합 구조를 적용해 모달 특성과 이상 패턴에 따라 서로 다른 Expert가 동적으로 선택되는 구조를 구현하였다. 이를 통해 다중 모달 입력은 물론 일부 모달 누락 상황에서도 동일한 모델이 일관된 방식으로 이상 여부를 판별할 수 있도록 설계되었으며, 최종적으로 정상/이상 이진 분류를 수행하는 전체 모델을 제공한다.

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
