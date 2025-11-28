# SKAB 시계열 이상 탐지 - TS2Vec 베이스라인

## 📋 전체 파이프라인 개요

이 프로젝트는 TS2Vec 기반 시계열 인코더를 사용하여 SKAB 데이터셋의 이상 탐지를 수행합니다.

```
anomaly-free (정상 데이터)  →  TS2Vec 학습  →  인코더 체크포인트
                                     ↓
                            정상 메모리 뱅크 구축
                                     ↓
other (테스트 데이터)  →  임베딩 추출  →  이상 점수 계산  →  평가
```

---

## 🎯 3가지 실행 모드

### 1️⃣ **train** 모드 - 인코더 학습
```bash
python sieun_baseline_train.py train \
  --data_root /data/jupyter/AGI/datasets/skab \
  --window 32 --stride 1 --batch 128 --epochs 80 \
  --out_dir ./outputs_skab
```

**동작:**
- `anomaly-free/*.csv` 폴더의 정상 데이터로 TS2Vec 인코더 학습
- Contrastive Self-Supervised Learning (데이터 증강 사용)
- 출력: `ts2vec_best.pt` (학습된 인코더 체크포인트)

**학습 데이터:**
- 정상 데이터만 사용 (`anomaly-free/anomaly-free.csv`)
- 윈도우 크기: 32, stride: 1
- 라벨은 무시 (자기지도학습)

---

### 2️⃣ **build_bank** 모드 - 정상 메모리 뱅크 구축
```bash
python sieun_baseline_train.py build_bank \
  --data_root /data/jupyter/AGI/datasets/skab \
  --window 32 --stride 1 \
  --ckpt ./outputs_skab/ts2vec_best.pt \
  --out_dir ./outputs_skab
```

**동작:**
- 학습된 인코더로 `anomaly-free` 데이터를 임베딩 Z로 변환
- 정상 데이터의 임베딩 분포를 저장 (이상 탐지 기준)

**출력 파일:**
- `memory_Z.npy` - 정상 데이터 임베딩 [N, 256]
- `memory_knn.pkl` - kNN 인덱스 (거리 기반 이상 점수)
- `memory_kde.pkl` - KDE 밀도 추정 (밀도 기반 이상 점수)
- `stats_mu.npy`, `stats_sd.npy` - 정규화 통계

---

### 3️⃣ **eval** 모드 - 테스트 데이터 평가 ⭐
```bash
python sieun_baseline_train.py eval \
  --data_root /data/jupyter/AGI/datasets/skab \
  --window 32 --stride 1 \
  --ckpt ./outputs_skab/ts2vec_best.pt \
  --bank_dir ./outputs_skab \
  --score_head kde \
  --smooth 5 \
  --out_dir ./outputs_skab
```

**동작:**
1. `other/`, `valve1/`, `valve2/` 폴더의 테스트 데이터 읽기 (이상치 포함)
2. 체크포인트로 임베딩 Z 추출
3. 정상 메모리 뱅크와 비교하여 이상 점수 계산
4. 메트릭 계산 및 결과 저장

**출력 파일:**
- 그룹별 디렉토리에 **윈도우별 개별 임베딩** 저장 ⭐
  - `other/{file}_{window_idx}.pt` - other 그룹 (예: `9_0000.pt`, `9_0001.pt`, ...)
  - `valve1/{file}_{window_idx}.pt` - valve1 그룹 (예: `0_0000.pt`, `0_0001.pt`, ...)
  - `valve2/{file}_{window_idx}.pt` - valve2 그룹 (예: `0_0000.pt`, `0_0001.pt`, ...)
  - 각 윈도우 파일 내용:
    - `embedding`: [256] 텐서 (단일 윈도우 임베딩)
    - `label`: int (윈도우 중간 라벨, 0=정상, 1=이상)
- `test_scores.csv` - 전체 점수, 라벨, 예측 결과 (통합)
- 콘솔에 메트릭 출력 (JSON 형식)

**테스트 데이터:**
- `other/*.csv` (14개 파일), `valve1/*.csv` (16개 파일), `valve2/*.csv` (4개 파일)
- 각 윈도우를 개별 샘플로 저장 (그룹별 디렉토리 분리)
- 이상치 포함 (라벨 컬럼: `anomaly`)

---

## 📁 SKAB 데이터셋 구조

```
/data/jupyter/AGI/datasets/skab/
├── anomaly-free/
│   └── anomaly-free.csv    # 정상 데이터 (train용)
│                           # 9,403 행
├── other/                  # 테스트 그룹 1
│   ├── 9.csv               # 14개 파일 (이상치 포함)
│   ├── 11.csv
│   ├── ...
│   └── 23.csv
├── valve1/                 # 테스트 그룹 2
│   ├── 0.csv               # 16개 파일 (이상치 포함)
│   ├── 1.csv
│   ├── ...
│   └── 15.csv
└── valve2/                 # 테스트 그룹 3
    ├── 0.csv               # 4개 파일 (이상치 포함)
    ├── 1.csv
    ├── 2.csv
    └── 3.csv
```

**CSV 구조:**
```
datetime;Accelerometer1RMS;Accelerometer2RMS;Current;Pressure;Temperature;Thermocouple;Voltage;Volume Flow RateRMS;anomaly;changepoint
```
- **특성:** 9개 센서 값 (Accelerometer1RMS ~ Volume Flow RateRMS)
- **라벨:** `anomaly` 컬럼 (0=정상, 1=이상)
- **자동 제거:** datetime, changepoint 컬럼

---

## 📊 평가 메트릭

```json
{
  "AUC_ROC": 0.565,              // ROC 곡선 아래 면적
  "AUC_PR": 0.405,               // Precision-Recall 곡선 아래 면적
  "F1_best": 0.560,              // argmax로 찾은 최적 F1 스코어
  "Threshold_F1_best": -467.4,   // 최적 F1 달성 임계값
  "F1_PA": 0.848,                // Point-Adjusted F1 (구간 단위 평가)
  "Threshold_KDE_cross": -451.5  // KDE 교차점 임계값 (참고용)
}
```

**F1-PA (Point-Adjusted F1):**
- 이상 구간을 하나의 단위로 취급
- 구간 내 하나의 포인트라도 탐지하면 해당 구간 전체를 성공으로 간주
- 실제 운영 환경에 더 적합한 메트릭

---

## 🔧 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `window` | 32 | 슬라이딩 윈도우 길이 |
| `stride` | 1 | 윈도우 이동 간격 (point-wise 평가) |
| `batch` | 128 | 배치 크기 |
| `epochs` | 80 | 학습 에포크 수 |
| `lr` | 1e-3 | 학습률 |
| `temp` | 0.1 | Contrastive loss temperature |
| `k_neighbors` | 10 | kNN 이웃 개수 |
| `smooth` | 5 | 이상 점수 스무딩 윈도우 |

---

## 🚀 전체 실행 순서

```bash
# conda 환경 활성화 (중요!)
conda activate agi_img_txt

cd /data/jupyter/AGI/encoders/uni_modal/series

# 1단계: 인코더 학습 
python sieun_baseline_train.py train \
  --data_root /data/jupyter/AGI/datasets/skab \
  --window 32 --stride 1 --batch 128 --epochs 80 \
  --out_dir ./outputs_skab

# 2단계: 메모리 뱅크 구축 
python sieun_baseline_train.py build_bank \
  --data_root /data/jupyter/AGI/datasets/skab \
  --window 32 --stride 1 \
  --ckpt ./outputs_skab/ts2vec_best.pt \
  --out_dir ./outputs_skab

# 3단계: 테스트 평가 및 임베딩 추출 
python sieun_baseline_train.py eval \
  --data_root /data/jupyter/AGI/datasets/skab \
  --window 32 --stride 1 \
  --ckpt ./outputs_skab/ts2vec_best.pt \
  --bank_dir ./outputs_skab \
  --score_head kde \
  --smooth 5 \
  --out_dir ./outputs_skab
```

---

## 📦 생성되는 파일들

```
outputs_skab/
├── ts2vec_best.pt              # ✅ 학습된 인코더 (train)
├── ts2vec_ckpt.pt              # 마지막 체크포인트
├── memory_Z.npy                # ✅ 정상 메모리 뱅크 [9370, 256] (build_bank)
├── memory_knn.pkl              # kNN 인덱스
├── memory_kde.pkl              # KDE 밀도 추정기
├── stats_mu.npy                # 정규화 평균
├── stats_sd.npy                # 정규화 표준편차
├── test_scores.csv             # 전체 점수 + 라벨 + 예측 (통합)
├── other/                      # ⭐ other 그룹 임베딩 (eval)
│   ├── 9_0000.pt               # 윈도우별 개별 샘플
│   ├── 9_0001.pt               # embedding + label 포함
│   ├── ...
│   ├── 11_0000.pt
│   └── ...
├── valve1/                     # ⭐ valve1 그룹 임베딩 (eval)
│   ├── 0_0000.pt
│   ├── 0_0001.pt
│   ├── ...
│   ├── 1_0000.pt
│   └── ...
└── valve2/                     # ⭐ valve2 그룹 임베딩 (eval)
    ├── 0_0000.pt
    ├── 0_0001.pt
    ├── ...
    └── 3_1115.pt
```

---

## 🎯 핵심 기능

### 1. 중간 라벨 사용 (Point-wise 평가)
- 윈도우 크기 32 → 16번째 포인트의 라벨 사용
- stride=1로 모든 포인트마다 평가
- 첫 번째 윈도우 [0:32] → 라벨 [15]
- 두 번째 윈도우 [1:33] → 라벨 [16]

### 2. 최적 F1 자동 선택 (argmax)
```python
# precision_recall_curve로 모든 임계값 탐색
ps, rs, ths = precision_recall_curve(labels, scores)
f1s = 2*ps*rs/(ps+rs+1e-8)
best_idx = np.argmax(f1s)  # ← 최고 F1 인덱스
best_thr = ths[best_idx]   # 최적 임계값
```

### 3. TS2Vec 아키텍처
```
입력 [B, 32, 9] 
  → TCN 블록 (dilation: 1, 2, 4, 1) 
  → Adaptive Pooling 
  → Projection Head 
  → 출력 [B, 256]
```

---

## 📈 임베딩 Z 사용 방법

### Python에서 개별 윈도우 로드:
```python
import torch
import numpy as np
from pathlib import Path

# 개별 윈도우 샘플 로드 (예: other/9.csv의 0번째 윈도우)
data = torch.load('outputs_skab/other/9_0000.pt')

print(f"임베딩 shape: {data['embedding'].shape}")  # torch.Size([256])
print(f"라벨: {data['label']}")                     # 0 or 1

# 임베딩과 라벨 추출
embedding = data['embedding']  # [256] 단일 윈도우 임베딩
label = data['label']          # 스칼라 (0=정상, 1=이상)

# NumPy로 변환
embedding_np = embedding.numpy()
```

### 그룹별로 모든 윈도우 로드:
```python
import torch
from pathlib import Path

output_dir = Path('outputs_skab')

# other 그룹 모든 윈도우 로드
other_dir = output_dir / 'other'
pt_files = sorted(other_dir.glob('*.pt'))

embeddings = []
labels = []

for pt_file in pt_files:
    data = torch.load(pt_file)
    embeddings.append(data['embedding'])  # [256]
    labels.append(data['label'])

# 텐서로 통합
embeddings_tensor = torch.stack(embeddings)  # [N, 256]
labels_tensor = torch.tensor(labels)         # [N]

print(f"Total windows: {len(embeddings)}")
print(f"Normal: {(labels_tensor == 0).sum()}, Anomaly: {(labels_tensor == 1).sum()}")
```

### 3개 그룹 모두 로드:
```python
import torch
from pathlib import Path

output_dir = Path('outputs_skab')
groups = ['other', 'valve1', 'valve2']

all_data = {g: {'embeddings': [], 'labels': []} for g in groups}

for group_name in groups:
    group_dir = output_dir / group_name
    if not group_dir.exists():
        continue
    
    pt_files = sorted(group_dir.glob('*.pt'))
    for pt_file in pt_files:
        data = torch.load(pt_file)
        all_data[group_name]['embeddings'].append(data['embedding'])
        all_data[group_name]['labels'].append(data['label'])
    
    # 그룹별 통합
    all_data[group_name]['embeddings'] = torch.stack(all_data[group_name]['embeddings'])
    all_data[group_name]['labels'] = torch.tensor(all_data[group_name]['labels'])
    
    print(f"{group_name}: {len(all_data[group_name]['embeddings'])} windows")

# 전체 통합
total_embeddings = torch.cat([all_data[g]['embeddings'] for g in groups], dim=0)
total_labels = torch.cat([all_data[g]['labels'] for g in groups], dim=0)
print(f"Total: {len(total_embeddings)} windows")
```

### 정상 메모리 뱅크 로드:
```python
import numpy as np

# 정상 데이터 임베딩
z_normal = np.load('outputs_skab/memory_Z.npy')
print(z_normal.shape)  # (9370, 256)
```

### 임베딩 활용 예시:
- **시각화:** t-SNE, UMAP으로 2D 투영 (파일별 색상 구분)
- **이상 탐지:** 새로운 이상 탐지 모델 학습 (SVM, Isolation Forest 등)
- **다른 모델:** 임베딩을 입력으로 사용 (Classifier, VAE, Transformer 등)
- **유사도 분석:** 코사인 유사도, 유클리디안 거리
- **파일별 분석:** 각 CSV 파일의 이상 패턴 개별 분석

---

## ⚠️ 주의사항

1. **conda 환경 필수:**
   ```bash
   conda activate agi_img_txt
   ```
   (base 환경에서는 torch import 에러 발생)

2. **GPU 메모리:**
   - 배치 크기가 크면 OOM 발생 가능
   - 메모리 부족 시 `--batch 64` 또는 `--batch 32`로 줄이기

3. **데이터 경로:**
   - `anomaly-free/` 또는 `train/` 폴더 필요
   - `other/` 또는 `test/` 폴더 필요

4. **실행 순서:**
   - 반드시 train → build_bank → eval 순서로 실행
   - 각 단계는 이전 단계의 출력 파일이 필요

---

## 📚 참고

**TS2Vec 논문:**
- Yue et al., "TS2Vec: Towards Universal Representation of Time Series" (AAAI 2022)

**주요 개념:**
- **Contrastive Learning:** 같은 샘플의 서로 다른 증강은 가깝게, 다른 샘플과는 멀게
- **Self-Supervised:** 라벨 없이 학습 (정상 데이터만 사용)
- **Memory Bank:** 정상 데이터의 임베딩 분포를 기준으로 이상 탐지

---

## 🐛 문제 해결

### torch import 에러
```bash
# 해결: agi_img_txt 환경 사용
conda activate agi_img_txt
```

### No train CSV files found
```bash
# 해결: 데이터 경로 확인
ls /data/jupyter/AGI/datasets/skab/anomaly-free/
ls /data/jupyter/AGI/datasets/skab/other/
```

### OOM (Out of Memory)
```bash
# 해결: 배치 크기 줄이기
python sieun_baseline_train.py train ... --batch 64
```

---

## ✅ 실행 결과 예시

```bash
# outputs_final/ 폴더 내용 (총 63MB)
11_embeddings.pt ~ 23_embeddings.pt, 9_embeddings.pt  # 14개 파일
memory_Z.npy, memory_knn.pkl, memory_kde.pkl
ts2vec_best.pt, ts2vec_ckpt.pt
test_scores.csv
```

**평가 메트릭 (80 에포크 학습):**
```json
{
  "AUC_ROC": 0.563,
  "AUC_PR": 0.420,
  "F1_best": 0.556,
  "F1_PA": 0.875
}
```

**개별 파일 예시 (9.csv):**
- 윈도우: 720개, 임베딩: [720, 256], 이상치: 179개 (24.9%)

---

**작성일:** 2025-10-28  
**버전:** 3.0 (윈도우별 개별 샘플 + 그룹별 디렉토리)  
**업데이트:** 3개 그룹(other, valve1, valve2) 지원, 윈도우별 개별 pt 저장

