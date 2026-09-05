# ? CIFAR-10 Classification with ResNet18: Transfer Learning, Ablation Study & XAI Verification

PyTorch 기반의 합성곱 신경망(ResNet18) 전이학습, 점진적 하이퍼파라미터 최적화(Ablation Study), 설명 가능한 AI(Grad-CAM) 및 시각화 신뢰성 검증(Sanity Check) 파이프라인입니다.

## ? 주요 구현 내용
- **Backbone & Architecture:** ImageNet 사전학습 ResNet18 백본 활용 및 CIFAR-10(10 Classes) 맞춤 완전연결계층(FC Layer) 재설계
- **Data Augmentation & Resolution:** RandomCrop, RandomHorizontalFlip 및 배경 편향 완화를 위한 `ColorJitter` 적용, 합성곱 특징 맵 공간 해상도 확보를 위한 224×224 업샘플링 파이프라인
- **Optimization & Scheduling:** AdamW Optimizer, CosineAnnealingLR 스케줄러를 통한 수렴 안정화 및 최적 체크포인트 자동 저장 (`weights/best_model_advanced.pth`)
- **Explainable AI (XAI):** ResNet18 최상단 합성곱 블록(`layer4[-1]`) 특징 맵 기반 Grad-CAM 의사결정 시각화
- **Model Sanity Check:** 논문 방법론(*Sanity Checks for Saliency Maps*, Adebayo et al., NeurIPS 2018)에 기반한 계층별 파라미터 무작위화 검증 수행

---

## ? 단계별 실험 결과 (Experiment Log)

| 실험 단계 | 반복 횟수 (Epochs) | Optimizer / Scheduler | 주요 적용 기법 | 최종 Loss | 검증 정확도 (Accuracy) | 핵심 의의 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | 3 | AdamW ($lr=10^{-3}$) | 기본 증강 | 0.73 | **77.86%** | 기본 학습 및 검증 파이프라인 구축 |
| **Extended Baseline** | 10 | AdamW ($lr=10^{-3}$) | 기본 증강 | 0.51 | **82.50%** | 장기 수렴 양상 확인 및 오답(Failure case) 분석 |
| **XAI Integration** | - | - | 224×224 업샘플링 | - | - | Layer 4 기반 고대비 Grad-CAM 시각화 연동 |
| **Optimized & Verified** | 12 | CosineAnnealingLR | ColorJitter + Sanity Check | **0.30** | **86.40%** | **손실값 진동 제어, 최고 성능 달성 (+3.9%p) 및 XAI 신뢰성 검증** |

![Ablation Comparison](results/ablation_compare.png)

### ? 최적화 실험 분석 (Ablation Study Insights)
1. **Cosine Annealing LR 스케줄러를 통한 수렴 안정화:**
   - 고정 학습률 사용 시 손실 함수 최저점 부근에서 보폭이 커 정확도가 요동치던 현상을 해결.
   - 학습률이 코사인 곡선을 그리며 $10^{-5}$까지 감속하면서 손실값이 0.30까지 정밀 수렴.
2. **ColorJitter 증강을 통한 배경 편향(Context Bias) 억제:**
   - 밝기·대비·채도를 무작위로 왜곡하여 모델이 저채도 배경 색감에 의존하던 지름길 학습(Shortcut Learning)을 차단, 일반화 성능 개선.

---

## ? Explainable AI: 베이스라인 vs 최적화 모델 의사결정 대조
최적화 기법 적용 전후 모델의 내부 주의 집중 영역(Attention) 변화를 비교하기 위해 정면 대조(Head-to-head Comparison)를 수행했습니다.

![Model Comparison](results/xai_model_comparison.png)

### ? 정성적 의사결정 교정 분석 (Qualitative Comparison)
- **사슴 (Target: deer):**
  - **Baseline (82.5%):** 사슴 본체 대신 좌측 나무 기둥 및 주변 풀숲 배경에 활성화가 강하게 분산되며 배경 편향으로 인해 전부 `cat`으로 오분류.
  - **Optimized (86.4%):** 배경 노이즈를 억제하고 **사슴의 뿔, 얼굴 윤곽, 등뼈 라인**에 활성화가 집중되며 정상 분류로 교정됨.

### ? 텐서 차원(Tensor Dimension) 및 아키텍처 관점의 고찰
- **`model.layer4[-1]` 타겟팅 이유:**
  - 초기 합성곱 층은 단순 엣지 중심이나, 최상단 합성곱 층은 고차원 의미 특징(Semantic Feature)을 포괄함.
  - `AdaptiveAvgPool2d`를 통과해 1차원 벡터($1 \times 1 \times 512$)로 공간 좌표가 손실되기 직전, 2차원 공간 정보($7 \times 7$)가 유지되는 마지막 계층이기 때문임.
- **224×224 업샘플링의 필요성:**
  - CIFAR-10 원본 해상도($32 \times 32$)는 계층을 통과하며 특징 맵이 $1 \times 1$ 크기로 축소되어 히트맵 해상도가 뭉개짐.
  - 입력을 $224 \times 224$로 업샘플링하여 최종 합성곱 계층의 공간 해상도를 확보함으로써 사물 국소 영역에 맺히는 고대비 히트맵 도출.

---

## ? XAI Model Sanity Check (가중치 무작위화 검증)
Saliency Map 기반 시각화 기법이 단순 엣지 검출기(Edge Detector)가 아닌 신경망 파라미터에 의존하는지(Faithfulness) 검증하는 **Model Parameter Randomization Test**를 진행했습니다.

![Sanity Check](results/sanity_check.png)

### ? 검증 결과 분석 (Sanity Check Passed)
- **실험 방식:** 정상 학습된 모델(`best_model_advanced.pth`)과 최상단 분류 블록(`layer4`, `fc`)을 Kaiming Normal 난수로 리셋한 모델의 Grad-CAM 활성화 맵을 대조.
- **검증 판정:**
  - **Cat / Ship 공통:** 학습된 정상 모델(중앙)은 사물의 형태적 핵심 부위에 초점을 형성한 반면, 파라미터가 파괴된 모델(우측)은 초점이 완전히 산란되어 무의미한 노이즈로 붕괴됨.
  - 이를 통해 추출된 히트맵이 신경망의 실제 학습된 가중치 상태에 직접적으로 의존함을 규명함.

---

## ? 실행 방법
```bash
# 1. 의존성 패키지 설치
pip install -r requirements.txt

# 2. 베이스라인 모델 학습
python3 main.py

# 3. 고도화 최적화 학습 및 Ablation Study (86.4% 달성)
python3 train_advanced.py

# 4. 베이스라인 vs 최적화 모델 간 Grad-CAM 대조 시각화
python3 cam_compare.py

# 5. XAI 모델 건전성 검증 (Sanity Check)
python3 sanity_check.py