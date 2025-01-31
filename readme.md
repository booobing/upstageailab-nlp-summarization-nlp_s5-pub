# NLP 대화 요약 모델 프로젝트
## Team

| ![김동완B](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | 
|            [김동완B](https://github.com/booobing)             |
|                            팀장, 개발 총괄                            | 


## 프로젝트 개요
대화 데이터를 요약하는 NLP 모델을 개발하고 최적화하는 프로젝트입니다. KoBART를 기반으로 하여 프롬프트 엔지니어링과 PPO 강화학습을 적용했습니다.

## 디렉토리 구조
```
.
├── code/
│   ├── baseline.ipynb        # 베이스라인 모델 구현
│   ├── config.yaml          # 설정 파일
│   ├── improving.py         # 모델 개선 스크립트
│   └── improving_all.py     # 통합 개선 스크립트
├── data/
│   ├── train.csv           # 학습 데이터
│   ├── val.csv            # 검증 데이터
│   └── test.csv           # 테스트 데이터
└── outputs/
    └── output_improved_final.csv  # 최종 결과 파일
```

## 주요 기능

### 1. 베이스라인 모델 (baseline.ipynb)
- KoBART 기반 요약 모델 구현
- ROUGE 점수 기반 성능 평가
- 기본 데이터 전처리 파이프라인

### 2. 모델 개선 (improving.py)
- 프롬프트 엔지니어링 구현
- SpellChecker 및 텍스트 정제
- Solar LLM API 통합
- 병렬 처리 최적화

### 3. 통합 개선 (improving_all.py)
- PPO 강화학습 구현
- 하이퍼파라미터 최적화
- 데이터 증강 기능
- 멀티 GPU 지원

## 설정 (config.yaml)
```yaml
model:
  name: "digit82/kobart-summarization"
  max_length: 128
  batch_size: 16
  learning_rate: 1e-5

training:
  epochs: 20
  gradient_accumulation_steps: 2
  warmup_ratio: 0.1

inference:
  temperature: 0.1
  top_p: 0.2
```

## 사용 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 베이스라인 모델 실행
```bash
jupyter notebook code/baseline.ipynb
```

### 3. 모델 개선 실행
```bash
python code/improving.py --config config.yaml
```

### 4. 전체 파이프라인 실행
```bash
python code/improving_all.py --config config.yaml
```

## 주요 성능 지표

| 모델 구성 | ROUGE 점수 | 특징 |
|----------|------------|------|
| 베이스라인 | 48.1088 | KoBART 기본 모델 |
| 프롬프트 개선 | 42.9535 | 객관적 문어체 적용 |
| Solar Pro + LLM | 44.5761 | 최적화된 파라미터 |

## 의존성 패키지
- transformers>=4.35.2
- torch>=1.13.0
- pandas
- numpy
- rouge-score
- accelerate>=0.26.0
- wandb
- pyyaml

## 주의사항
1. GPU 메모리 관리
   - 배치 사이즈 조정 필요
   - Mixed Precision 사용 권장

2. API 사용
   - Solar LLM API 키 필요
   - Rate limit 고려 필요

3. 데이터 처리
   - 대용량 데이터 처리 시 병렬화 권장
   - 중간 결과 저장 필요

## 라이센스
MIT License

## 참고사항
- 프로젝트 관련 상세 내용은 review/nlp_competition_review.md 참고
- 실험 기록은 review/nlp경진대회_수기록.txt 참고 

## 코드 구조 상세 설명

### 📁 code/
#### baseline.ipynb
- 베이스라인 KoBART 모델 구현
- 주요 기능:
  - 데이터 로딩 및 전처리
  - 모델 학습 및 평가
  - ROUGE 스코어 계산
  - 체크포인트 저장/로드
- 핵심 컴포넌트:
  ```python
  class DialogueDataset(Dataset):
      # 대화 데이터셋 처리
  
  class CustomTrainer(Trainer):
      # ROUGE 스코어 기반 평가 로직
  ```

#### improving.py
- 모델 성능 개선을 위한 스크립트
- 주요 기능:
  - SpellChecker 기반 텍스트 정제
  - 프롬프트 엔지니어링
  - Solar LLM API 통합
  - 병렬 처리 최적화
- 핵심 컴포넌트:
  ```python
  class TextProcessor:
      # 텍스트 전처리 및 정제
  
  class PromptManager:
      # 프롬프트 템플릿 관리
  ```

#### improving_all.py
- 통합 개선 파이프라인
- 주요 기능:
  - PPO 강화학습 구현
  - 멀티 GPU 지원
  - wandb 통합
  - 자동 하이퍼파라미터 튜닝
- 핵심 컴포넌트:
  ```python
  class PPOTrainer:
      # PPO 알고리즘 구현
  
  class MultiGPUManager:
      # GPU 리소스 관리
  ```

#### config.yaml
- 모델 및 학습 설정 파일
- 주요 섹션:
  ```yaml
  model:
    # 모델 기본 설정
  
  training:
    # 학습 관련 파라미터
  
  inference:
    # 추론 관련 설정
  
  preprocessing:
    # 전처리 관련 설정
  ```

#### utils/
- 유틸리티 함수 모음
  - `metrics.py`: ROUGE 스코어 계산
  - `data_utils.py`: 데이터 처리 유틸리티
  - `model_utils.py`: 모델 관련 유틸리티

#### scripts/
- 실행 스크립트 모음
  - `train.sh`: 학습 실행
  - `evaluate.sh`: 평가 실행
  - `preprocess.sh`: 전처리 실행

#### solar_api.ipynb
- Solar Chat API를 활용한 대화 요약 구현
- 주요 기능:
  - Solar API 연동 및 프롬프트 엔지니어링
  - 배치 처리 및 Rate Limit 관리
  - ROUGE 스코어 평가

#### test_autolearning.py
- 자동 학습 최적화 구현
- 주요 기능:
  - PPO(Proximal Policy Optimization) 구현
  - 가중치 검증 및 초기화
  - 멀티 GPU 지원
  - wandb 통합 모니터링

#### test_failed.py / test_failed2.py
- 실패한 실험들의 코드 보관
- 시도된 접근법:
  - 체크포인트 평가 시스템
  - 다양한 데이터 전처리 방식
  - 모델 구조 변경 실험

#### test_new.py
- 프롬프트 엔지니어링 개선 버전
- 특징:
  - [INST] 태그를 활용한 프롬프트 구조화
  - 컨텍스트 기반 요약 강화
  - 512 토큰 길이 지원

#### test_nonprompt.py
- 프롬프트 없는 베이스라인 모델
- 특징:
  - 순수 KoBART 성능 테스트
  - 기본 전처리만 적용
  - 비교 실험용 코드

#### test_wandb.py
- Weights & Biases 통합 버전
- 주요 기능:
  - 실험 로깅 및 시각화
  - 하이퍼파라미터 추적
  - 성능 메트릭 모니터링

#### test.py / test2.py
- 메인 실험 코드
- 특징:
  - 기본 학습 및 추론 파이프라인
  - 다양한 설정 실험
  - 체크포인트 관리

## 실험 버전별 주요 차이점

| 버전 | 특징 | ROUGE 점수 | 비고 |
|-----|------|------------|------|
| solar_api | API 기반 | 39.9742 | API 속도 제한 있음 |
| test_new | 프롬프트 개선 | 42.9535 | 안정적인 성능 |
| test_nonprompt | 순수 베이스라인 | 48.1088 | 가장 높은 성능 |
| test_autolearning | PPO 적용 | 44.5761 | 학습 시간 긺 |

## 실험 환경 설정

### Solar API 사용
```bash
python code/solar_api.py --api-key "your_key" --temperature 0.1 --top-p 0.2
```

### 자동 학습 실행
```bash
python code/test_autolearning.py --multi-gpu --ppo
```

### Weights & Biases 모니터링
```bash
python code/test_wandb.py --project "nlp_stage" --entity "your_entity"
```

## 실행 예시

### 베이스라인 모델 학습
```bash
bash scripts/train.sh --model baseline
```

### 프롬프트 엔지니어링 적용
```bash
python code/improving.py --prompt-template "객관적 관점에서 다음 대화를 요약하시오:"
```

### PPO 강화학습 실행
```bash
python code/improving_all.py --ppo --epochs 20 --reward-scale 0.1
```

### 멀티 GPU 학습
```bash
python code/improving_all.py --multi-gpu --gpu-ids 0,1,2,3
```

## 데이터 처리 파이프라인

1. 전처리 단계
```python
# 텍스트 정제
text_processor = TextProcessor(config)
cleaned_data = text_processor.clean_text(raw_data)

# 프롬프트 적용
prompt_manager = PromptManager(config)
processed_data = prompt_manager.apply_prompt(cleaned_data)
```

2. 학습 단계
```python
# 모델 학습
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
```

3. 평가 단계
```python
# ROUGE 스코어 계산
metrics = compute_metrics(predictions, references)
wandb.log(metrics)
```
