# IMDb 감성 분류 실험: Full Fine-Tuning vs LoRA / Prefix Tuning


## 개요

본 프로젝트는 IMDb 이진 감성 분류 데이터셋을 기반으로, 사전학습 언어모델에 대한 다양한 Fine-Tuning 기법의 성능 및 효율성을 비교하고자 수행되었습니다. 실험은 다음과 같이 두 파트로 나뉘어 진행되었습니다.

- **Part 1**: DistilBERT 모델 기반 Full Fine-Tuning vs LoRA (Low-Rank Adaptation)
- **Part 2**: BERT-base 모델 기반 Full Fine-Tuning vs Prefix Tuning

두 실험 모두 동일한 데이터셋과 유사한 학습 조건에서 진행되었으며, 정확도, 손실값, 파라미터 수, 훈련 시간, GPU 메모리 사용량을 기준으로 비교 분석하였습니다.

---

## 실험 환경

- **모델**:
  - Part 1: `distilbert-base-uncased`
  - Part 2: `bert-base-uncased`
- **데이터**: IMDb 감성 분류 (긍정/부정 이진 분류, `sentences`, `labels` 컬럼 포함 CSV)
- **입력 길이**: 최대 256 토큰 (`padding='max_length'`, `truncation=True`)
- **하드웨어**: Google Colab Pro (NVIDIA A100 GPU)
- **프레임워크**: Huggingface Transformers, Datasets, PEFT, Accelerate
- **공통 학습 조건**:
  - Epochs: 3
  - Batch Size: 학습 16 / 평가 64
  - FP16 mixed precision: 사용

---

## Part 1: Full Fine-Tuning vs LoRA (DistilBERT)

| 항목 | Full Fine-Tuning | LoRA Fine-Tuning |
|------|------------------|------------------|
| 학습 파라미터 | 전체 학습 | `q_lin`, `v_lin` 모듈만 학습 (r=8, alpha=16, dropout=0.05) |
| 정확도 | 92.2% | 89.9% |
| 훈련 시간 | 약 4분 30초 | 4분 24초 |
| GPU 메모리 사용량 | 약 3340MB | 약 2588MB (약 22% 절감) |

**해석**:
- LoRA는 학습 범위를 제한함으로써 메모리 사용량을 줄이고 학습 속도를 약간 개선함.
- 정확도는 Full Fine-Tuning이 약 2.3% 더 높았으나, 효율성과 단순화 면에서는 LoRA가 유리함.

---

## Part 2: Full Fine-Tuning vs Prefix Tuning (BERT-base)

| 항목 | Full Fine-Tuning | Prefix Tuning |
|------|------------------|----------------|
| 학습 파라미터 수 | 약 109M | 약 9.8M |
| 정확도 | 93.09% | 90.79% |
| 손실값 | 0.3227 | 0.2374 |

**해석**:
- Prefix Tuning은 전체 파라미터 대비 약 10분의 1 이하만 학습하며도 높은 성능 유지.
- 손실값은 오히려 Prefix Tuning이 낮았으며, 이는 모델의 예측 안정성이 높았음을 시사.
- 전체 파라미터를 학습하는 Full 방식은 여전히 최고 성능 달성에 유리하나, 자원 소모가 큼.

---

## 종합 비교

| 비교 항목 | Full Fine-Tuning | LoRA / Prefix Tuning |
|-----------|------------------|------------------------|
| 파라미터 수 | 매우 큼 (전체) | 제한적 (선택 모듈만) |
| 정확도 | 높음 | 약간 낮음 (~2% 이내) |
| 자원 효율 | 높지 않음 | 메모리, 시간 절약 |
| 적용 유연성 | 일반적 | 특정 구조 필요 |

**결론**: 정확도가 중요한 상황에서는 Full Fine-Tuning이 적합하며, 리소스 제한 환경에서는 LoRA 또는 Prefix Tuning이 효과적인 대안이 될 수 있음. 두 방식은 상호보완적 관계로 해석될 수 있음.

---

## 느낀 점

이번 실험은 단순한 성능 비교를 넘어서, 모델 경량화와 실용성이라는 관점에서 다양한 튜닝 기법을 직접 적용하고 평가해볼 수 있었던 경험이었다. 특히 LoRA와 Prefix Tuning은 기존 Fine-Tuning 코드 구조를 크게 변경하지 않고도 적용할 수 있었으며, 자원 효율성과 구현 편의성 측면에서 유의미한 가능성을 보여주었다.

버전 충돌 등 실험 환경 설정에 시간이 소요되었지만, Huggingface와 PEFT 생태계의 활용법을 실전에서 익히는 계기가 되었다. 향후에는 Prompt Tuning이나 Adapter 등의 기법에 대해서도 비교 실험을 진행할 계획이다.

---

## 디렉토리 구조

- `part2_lora/`: DistilBERT + LoRA 실험 코드 및 결과
- `part3_prefix/`: BERT + Prefix Tuning 실험 코드 및 결과

각 디렉토리에는 학습 스크립트, 설정 파일, 로그 출력 결과가 포함되어 있습니다.
