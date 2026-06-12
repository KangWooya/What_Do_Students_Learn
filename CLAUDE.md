# WDSL 프로젝트 가이드

## 논문 개요

**제목:** What Do Students Learn? A Feature-Level Analysis of Dark Knowledge  
**저자:** Seungu Kang, Songkuk Kim (Yonsei University, Seoul, Republic of Korea)  
**데이터셋:** CIFAR-100 (주요), CIFAR-10 (IT 프레임워크 분석)  
**모델:** ResNet-18, ResNet-34, ResNet-50 (Student), ResNet-152 (Teacher)

### 핵심 기여

1. **Interaction Tensor(IT) 기반 피처 수준 분석**: KD가 학생 모델에서 저빈도·샘플 특화 피처를 제거하고 고빈도·재사용 가능 피처를 강화하는 정규화 효과가 있음을 정량적으로 규명
2. **Confusion = Dark Knowledge 발견**: 모델의 혼동 행렬(confusion matrix)이 교사의 소프트 타겟과 유사한 클래스 간 유사도 구조를 인코딩함을 실증 (Pearson r ≈ 0.87, cosine ≈ 0.78)
3. **Confusion Distillation(CD) 방법 제안**: 교사 모델 없이 자신의 혼동 패턴을 동적 소프트 타겟으로 활용하는 자기 증류 방법. CIFAR-100에서 CS-KD, PS-KD 대비 약 1.2% 향상

---

## 카메라 레디 보강 사항 (REVIEWS.md 기반)

리뷰어 3인 및 메타 리뷰어 공통 지적 사항:

| 우선순위 | 보강 항목 | 해당 리뷰어 |
|---|---|---|
| **높음** | ImageNet-1K 실험 추가 (스케일 검증) | R1, R3, Meta |
| **높음** | 최신 비교 방법 추가: FitNets, ReviewKD 등 feature-based 방법 | R3, Meta |
| **중간** | CIFAR-100 외 추가 데이터셋 평가 | R1, R2, Meta |
| **중간** | ResNet 외 아키텍처 평가 (VGG, DenseNet 등) | R2, Meta |
| **중간** | IT 프레임워크 실용적 설명 개선 (대형 모델 확장 방법) | R3 |
| **낮음** | confusion이 dark knowledge 근사하는 이론적 근거 강화 | R1 |
| **낮음** | 전환 스케줄/손실 비율 민감도 분석 추가 | R1 |
| **낮음** | 클래스 불균형 상황에서의 CD 동작 논의 | Meta |

---

## 디렉토리 구조

```
WDSL/
├── Confusion_distillation/     # CD 방법 구현 및 분석 코드
├── Interaction_Tensor/         # IT 프레임워크 구현 및 분석 코드
├── CLAUDE.md                   # 이 파일
├── REVIEWS.md                  # 심사 의견 (카메라 레디 참고)
└── What_Do_Students_Learn.pdf  # 논문 원고
```

---

## Confusion_distillation/ 폴더

Windows 환경에서 작업된 폴더. 내부 경로 및 실행 명령어가 Linux 환경과 다를 수 있으므로 실행 전 경로 확인 필요.

### 핵심 노트북

| 파일 | 역할 | 논문 연관 | 카메라 레디 필요 |
|---|---|---|---|
| [Confusion_distillation.ipynb](Confusion_distillation/Confusion_distillation.ipynb) | CD 방법 핵심 구현. EMA 기반 2-phase training, CIFAR-100 학습 및 t-SNE 시각화 | Table 2 결과 생성, 피처 시각화 | **필수** |
| [Confusion_Distill_IT_analysis.ipynb](Confusion_distillation/Confusion_Distill_IT_analysis.ipynb) | 20개 CD 모델의 IT 분석. confusion matrix와 교사 소프트 타겟 간 상관관계 검증 | Section 4 피처 분석 핵심 | **필수** |
| [Self_Distill_IT_analysis.ipynb](Confusion_distillation/Self_Distill_IT_analysis.ipynb) | 자기증류 모델의 IT 분석. 베이스라인 대비 피처 재사용성 변화 분석 | Section 4 비교 분석 | **필수** |
| [Compare_models.ipynb](Confusion_distillation/Compare_models.ipynb) | CS-KD / PS-KD / CD 정확도 비교 (mean ± std, 3 runs). 이종 구현체 간 모델 로딩 처리 | Table 2 직접 생성 | **필수** |
| [Exploring_Confusion.ipynb](Confusion_distillation/Exploring_Confusion.ipynb) | 초기 탐색용. confusion matrix 히트맵, 차이 행렬(seismic colormap) 시각화 | 논문 직접 사용 X (탐색 과정) | 불필요 → archive 예정 |

### 학습 프레임워크 (서브디렉토리)

| 폴더 | 역할 | 논문 연관 | 카메라 레디 필요 |
|---|---|---|---|
| [pytorch-cifar100/](Confusion_distillation/pytorch-cifar100/) | 베이스라인 모델 및 KD 모델 학습. `train.py`, `train_kd.py` 포함 | Teacher, Basemodel, KDmodel 학습에 사용 | **필수** (재현성) |
| [cs-kd/](Confusion_distillation/cs-kd/) | CS-KD 방법 구현 (비교 대상). ResNet-18/34/50 CIFAR-100 결과 보유 | Table 2 CS-KD 결과 | 필요 (비교 재현) |
| [PS-KD-Pytorch/](Confusion_distillation/PS-KD-Pytorch/) | PS-KD 방법 구현 (비교 대상). CIFAR-100 데이터셋 포함 | Table 2 PS-KD 결과 | 필요 (비교 재현) |

### 모델 체크포인트

| 폴더/파일 | 내용 | 논문 연관 | 비고 |
|---|---|---|---|
| [Teachers/](Confusion_distillation/Teachers/) | ResNet-18 교사 모델 20개 (t1.pth~t20.pth, 각 ~43MB) | IT 분석 앙상블 | **필수** |
| [Basemodels/](Confusion_distillation/Basemodels/) | ResNet-18 베이스라인 모델 20개 (bm1.pth~bm20.pth) | IT 분석 기준선 | **필수** |
| [CDmodels/](Confusion_distillation/CDmodels/) | Confusion Distillation 모델 20개 (cd1.pth~cd20.pth) | IT 분석 CD 모델 | **필수** |
| [KDmodels/](Confusion_distillation/KDmodels/) | 교사-학생 KD로 학습된 스튜던트 모델 | IT 분석 KD 모델 | **필수** |
| [interaction_tensor_B_SCD_model.pt](Confusion_distillation/interaction_tensor_B_SCD_model.pt) | 베이스라인 + 자기증류 모델 IT 텐서 (896MB) | Section 4 분석 데이터 | **필수** |
| [interaction_tensor_CD_models.pt](Confusion_distillation/interaction_tensor_CD_models.pt) | CD 모델 IT 텐서 (1.4GB) | Section 4 분석 데이터 | **필수** |

### 기타

| 파일 | 역할 | 비고 |
|---|---|---|
| [run_command.txt](Confusion_distillation/run_command.txt) | 학습 명령어 모음 (PowerShell 형식) | Linux 환경에서 실행 시 `python` 명령어 및 경로 수정 필요 |

---

## Interaction_Tensor/ 폴더

Windows 환경에서 작업된 폴더. 내부 경로 확인 필요.

### 핵심 노트북

| 파일 | 역할 | 논문 연관 | 카메라 레디 필요 |
|---|---|---|---|
| [Interaction_Tensor.ipynb](Interaction_Tensor/Interaction_Tensor.ipynb) | IT 프레임워크 핵심 구현. CIFAR-10에서 20개 ResNet-18 독립 학습 후 피처 상호작용 분석 | Section 3 IT 프레임워크 설명 | **필수** |
| [KD_analysis_via_IT.ipynb](Interaction_Tensor/KD_analysis_via_IT.ipynb) | KD 효과의 IT 분석. Teacher(RN152) → Student(RN18) 피처 빈도 및 재사용성 변화 분석 | Section 4 핵심 분석 결과 | **필수** |
| [Self_Distill_IT_analysis.ipynb](Interaction_Tensor/Self_Distill_IT_analysis.ipynb) | 자기증류 모델 IT 분석. 베이스라인 대비 피처 패턴 변화 검증 | Section 4 비교 | **필수** |
| [KD_model.ipynb](Interaction_Tensor/KD_model.ipynb) | KD 모델 학습 및 저장 | IT 분석용 모델 생성 | 필요 (재현성) |
| [Exploring_Confusion.ipynb](Interaction_Tensor/Exploring_Confusion.ipynb) | 초기 탐색용 confusion 분석 | 탐색 과정 | 불필요 → archive 예정 |
| [Test_Unbalance.ipynb](Interaction_Tensor/Test_Unbalance.ipynb) | 불균형 데이터 실험 탐색 | 탐색 과정 | 불필요 → archive 예정 |
| [TestResnet.ipynb](Interaction_Tensor/TestResnet.ipynb) | ResNet 동작 테스트 | 탐색 과정 | 불필요 → archive 예정 |
| [tb2.ipynb](Interaction_Tensor/tb2.ipynb) | 디버깅/탐색용 스크래치 | 탐색 과정 | 불필요 → archive 예정 |

### IT 텐서 데이터 파일

| 파일 | 내용 | 논문 연관 | 비고 |
|---|---|---|---|
| [interaction_tensor.pt](Interaction_Tensor/interaction_tensor.pt) | 베이스라인 20개 모델 IT 텐서 (396MB) | Section 3 분석 데이터 | **필수** |
| [interaction_tensor_KD_models.pt](Interaction_Tensor/interaction_tensor_KD_models.pt) | KD 모델 IT 텐서 (1.8GB) | Section 4 핵심 데이터 | **필수** |
| [interaction_tensor_KD_models_20teacher.pt](Interaction_Tensor/interaction_tensor_KD_models_20teacher.pt) | 20개 교사 모델 IT 텐서 (1.7GB) | Section 4 분석 데이터 | **필수** |
| [interaction_tensor_KD_models_before_avgpool.pt](Interaction_Tensor/interaction_tensor_KD_models_before_avgpool.pt) | avgpool 이전 레이어 IT 텐서 (1.0GB) | 추가 분석용 | 사용 여부 검토 필요 |

---

## 코드 정리 계획 (단계별)

### Phase 1: 현재 단계 — 핵심 코드 식별 (문서화)
- [x] 각 파일 역할 및 논문 연관성 파악 → 이 CLAUDE.md

### Phase 2: 경로 및 환경 적응 ✅ 완료 (2026-04-01)
- [x] CIFAR-100 심볼릭 링크 생성: `/home/seungu/mycode/Data/cifar-100-python` → `PS-KD-Pytorch/data/cifar-100-python`
- [x] `pytorch-cifar100/utils.py` datapath 수정 → `/home/seungu/mycode/Data`
- [x] 전체 노트북 13개 `data_path` 수정 (`C:/Users/ksw00/mycode/data` → `/home/seungu/mycode/Data`)
- [x] `Compare_models.ipynb`, `KD_analysis_via_IT.ipynb` 모델/체크포인트 경로 수정
- [x] `run_command.sh` 생성 (Linux bash 버전, 원본 `run_command.txt`는 PowerShell 참고용으로 유지)
- [ ] 핵심 노트북 실행 검증 (셀 단위 실행 확인) — 카메라 레디 실험 시 수행

### Phase 3: 코드 통합 (추후)
- [ ] 탐색용 노트북 `archive/` 폴더로 이동
  - `Confusion_distillation/Exploring_Confusion.ipynb`
  - `Interaction_Tensor/Exploring_Confusion.ipynb`
  - `Interaction_Tensor/Test_Unbalance.ipynb`
  - `Interaction_Tensor/TestResnet.ipynb`
  - `Interaction_Tensor/tb2.ipynb`
- [ ] 중복 구현 정리 (CD 학습 코드를 단일 스크립트로 통합 검토)
- [ ] Linux 환경 재현 가이드 작성 (requirements, 학습 명령어)

### Phase 4: 카메라 레디 추가 실험 (확정 범위, 2026-04-01)

**결정 사항:** Tiny-ImageNet + DenseNet-121 + BYOT 비교

#### 4-1. 추가 아키텍처: DenseNet-121 on CIFAR-100
- [ ] Baseline DenseNet-121 학습 (`-net densenet121` 플래그만 추가, 이미 구현됨)
- [ ] CD DenseNet-121 학습 (`train_cd.py -net densenet121`)
- [ ] CS-KD DenseNet-121 학습 (`cs-kd/train.py --model CIFAR_DenseNet121` — 모델 등록 여부 확인 필요)
- [ ] PS-KD DenseNet-121 학습 (`PS-KD-Pytorch/` — DenseNet 지원 여부 확인 필요)
- [ ] Table 2에 DenseNet-121 열 추가

#### 4-2. 추가 데이터셋: Tiny-ImageNet (200 class, 100k, 64×64)
- [ ] Tiny-ImageNet 다운로드 → `/home/seungu/mycode/Data/tiny-imagenet-200/`
  ```bash
  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P /home/seungu/mycode/Data/
  unzip /home/seungu/mycode/Data/tiny-imagenet-200.zip -d /home/seungu/mycode/Data/
  ```
- [ ] `pytorch-cifar100/` 또는 별도 스크립트에 Tiny-ImageNet 데이터로더 추가
  - 입력 크기 64×64 대응 (CIFAR는 32×32)
  - `num_class=200` 변경 필요
- [ ] Baseline / CD / BYOT ResNet-18 학습 on Tiny-ImageNet (최소 구성)
- [ ] 새 Table (Tiny-ImageNet 결과) 추가

#### 4-3. 추가 비교 방법: BYOT (Be Your Own Teacher, 2021)
- [ ] BYOT 구현체 준비 (공개 코드 참고: auxiliary branch 방식)
  - 중간 레이어에 보조 분류기(auxiliary classifier) 추가 → 자기 증류
  - CIFAR-100 기준 ResNet-18, 34, 50 + DenseNet-121 실험
- [ ] Table 2 및 Tiny-ImageNet 테이블에 BYOT 행 추가

#### 4-4. 페이지 제한 대응 (현재 15페이지 꽉참)
- [ ] 기존 Table 2를 압축 또는 일부 결과 Appendix로 이동 검토
- [ ] IT 분석 그림 일부 압축 또는 병합 검토
- [ ] Tiny-ImageNet 결과를 간결한 보조 테이블로 구성

**가용 리소스 요약:**
| 항목 | 상태 |
|---|---|
| DenseNet-121 모델 구현 | ✅ `pytorch-cifar100/models/densenet.py`, `get_network()`에 등록됨 |
| Tiny-ImageNet 데이터 | ❌ 다운로드 필요 |
| BYOT 구현 | ❌ 신규 구현 필요 |
| CS-KD DenseNet 지원 | ✅ `--model CIFAR_DenseNet121` 플래그로 바로 사용 가능 |
| PS-KD DenseNet 지원 | ✅ `--classifier_type DenseNet121` 플래그로 바로 사용 가능 |

---

## 주요 하이퍼파라미터 (논문 기준)

| 파라미터 | 값 | 용도 |
|---|---|---|
| Temperature T | 2.0 | Soft label 온도 |
| soft_w : ce_w | 0.7 : 0.3 (학습) / 0.85 : 0.15 (평가 최적) | 손실 가중치 |
| EMA momentum μ | 0.9 | Confusion matrix 스무딩 |
| smoothing | 0.1 | 초기 스무딩 행렬 S |
| 전환 스케줄 | 3:3:6:3:15 (300 epoch 기준) | Phase 1→2 전환 |
| 학습률 warm-up | 선형 | SGD + momentum |
| Weight decay | 5×10⁻⁴ | 정규화 |

---

## DenseNet-121 CIFAR-100 실험 체크포인트 (카메라 레디, 2026-04-08)

`BASE = Confusion_distillation/`

### Baseline (200 epoch, `pytorch-cifar100/conf/global_settings.py` EPOCH=200)

| Run | 체크포인트 경로 | 완료 시각 |
|---|---|---|
| Run 1 | `pytorch-cifar100/checkpoint/densenet121/06_April_2026_17h/` | 2026-04-06 21:09 |
| Run 2 | `pytorch-cifar100/checkpoint/densenet121/06_April_2026_21h/` | 2026-04-07 00:19 |
| Run 3 | `pytorch-cifar100/checkpoint/densenet121/07_April_2026_00h/` | 2026-04-07 02:31 |

- best weight: `densenet121-{epoch}-best.pth`, 마지막 weight: `densenet121-200-regular.pth`

### CS-KD (200 epoch)

| Run | 체크포인트 경로 | Best Acc |
|---|---|---|
| Run 1 | `cs-kd/results/cifar100/CIFAR_DenseNet121/densenet121_run1/ckpt.t7` | 78.59% |
| Run 2 | `cs-kd/results/cifar100/CIFAR_DenseNet121/densenet121_run2/ckpt.t7` | 77.86% |
| Run 3 | `cs-kd/results/cifar100/CIFAR_DenseNet121/densenet121_run3/ckpt.t7` | 78.20% |

- Run 1은 고스트 프로세스(2026-04-06 17:00~23:14)로 학습된 결과이나 200 epoch 완주 확인됨

### PS-KD (300 epoch, `--lr_decay_schedule 150 225`)

| Run | 체크포인트 경로 |
|---|---|
| Run 1 | `PS-KD-Pytorch/runs/pskd_densenet121_cifar100/cifar100_DenseNet121_PSKD_True_2026-4-7-9-8-10/model/checkpoint_best.pth` |
| Run 2 | `PS-KD-Pytorch/runs/pskd_densenet121_cifar100/cifar100_DenseNet121_PSKD_True_2026-4-7-12-25-33/model/checkpoint_best.pth` |
| Run 3 | `PS-KD-Pytorch/runs/pskd_densenet121_cifar100/cifar100_DenseNet121_PSKD_True_2026-4-7-15-46-5/model/checkpoint_best.pth` |

- val acc ~80.6% (3회 평균)

### CD (300 epoch, `global_settings.py` EPOCH=300, MILESTONES=[90,180,240])

> 이전 200 epoch 결과(`08_April_2026_09h/`, `11h/`, `14h/`)는 폐기 대상

| Run | 체크포인트 경로 | Best Acc | 완료 시각 |
|---|---|---|---|
| Run 1 | `pytorch-cifar100/checkpoint/densenet121/08_April_2026_16h/` | 79.51% | 2026-04-08 20:12 |
| Run 2 | `pytorch-cifar100/checkpoint/densenet121/08_April_2026_20h/` | 79.03% | 2026-04-08 23:35 |
| Run 3 | `pytorch-cifar100/checkpoint/densenet121/08_April_2026_23h/` | 79.14% | 2026-04-09 02:57 |

- best weight: `densenet121-{epoch}-best.pth`
- `--transition_epoch 30 30 60 30 150` (따옴표 없이 전달해야 파싱 오류 없음)
- 3회 평균 val acc: **79.23%**

---

## 환경 주의사항

- `Confusion_distillation/`와 `Interaction_Tensor/` 모두 Windows에서 이전된 폴더
- `run_command.txt`의 명령어는 PowerShell 형식 (예: `python train.py` → Linux에서 동일하나 경로 구분자 `\` → `/` 변환 필요)
- 각 서브폴더(cs-kd, PS-KD-Pytorch, pytorch-cifar100)는 독립적인 git 저장소
- 대용량 `.pt` 텐서 파일(총 ~5.9GB)은 모델 재학습 없이 분석 재현에 필수
