# BART 모델을 활용한 텍스트 요약 시스템 개발 및 개선 과정

## 1. 개발 목표
- BART 모델을 활용한 텍스트 요약 시스템 구현
- PPO(Proximal Policy Optimization) 알고리즘을 통한 모델 성능 개선
- Rouge 점수를 통한 요약 품질 평가 및 최적화

## 2. 주요 문제점과 해결 과정

### 2.1 초기 구현 시 발생한 문제들

#### a. Trainer 관련 오류 
ImportError: Using the Trainer with PyTorch requires accelerate>=0.26.0

- 원인: transformers의 Trainer 클래스 사용을 위한 accelerate 라이브러리 버전 미달
- 해결: accelerate 라이브러리 업그레이드 수행

bash
pip install --upgrade accelerate


#### b. num_labels 관련 경고

You passed along num_labels=3 with an incompatible id to label map: {'0': 'NEGATIVE', '1': 'POSITIVE'}

- 원인: 요약 작업에 불필요한 num_labels 설정 존재
- 해결: config 파일에서 num_labels 설정 제거

#### c. encoder 접근 오류

AttributeError: 'BartForConditionalGeneration' object has no attribute 'encoder'

- 원인: BartForConditionalGeneration 모델의 잘못된 encoder 접근 방식
- 해결: generate_model.encoder를 generate_model.model.encoder로 수정

### 2.2 PPO 학습 과정에서의 문제점

#### a. 텐서 크기 불일치 오류

RuntimeError: The size of tensor a (23) must match the size of tensor b (16) at non-singleton dimension 1

- 원인: ratios와 advantages 텐서의 시퀀스 길이 불일치
- 해결:
  - max_length 고정을 통한 시퀀스 길이 통일
  - 패딩 적용으로 텐서 크기 맞춤
  - advantages 텐서 차원 확장

## 3. 주요 개선 사항

### 3.1 메모리 최적화
- 배치 사이즈 조정 (32 → 16)
- 그라디언트 누적 steps 도입 (2)
- Gradient Checkpointing 활성화
- Mixed Precision (FP16) 사용

### 3.2 모델 구조 개선
- encoder 파라미터 고정으로 학습 효율성 향상
- 최대 시퀀스 길이 제한으로 메모리 사용량 감소
- DataLoader workers 수 최적화

### 3.3 학습 프로세스 개선
- Rouge 점수 기반 조기 종료 구현
- 최고 성능 모델 자동 저장 기능
- 학습 과정 로깅 시스템 구축

## 4. 시행착오 및 교훈

### 4.1 모델 구조 이해의 중요성
- BART 모델의 내부 구조 이해 부족으로 인한 초기 오류 발생
- 정확한 모델 구조 파악의 중요성 인식

### 4.2 메모리 관리의 중요성
- 대규모 언어 모델 학습 시 메모리 관리의 중요성 확인
- 다양한 메모리 최적화 기법 적용 필요성 인식

### 4.3 텐서 연산 디버깅
- 텐서 크기 불일치로 인한 문제 해결 과정에서 디버깅의 중요성 인식
- 텐서 형태 출력을 통한 효과적인 문제 진단 방법 학습

## 5. 최종 개선된 시스템 구조

### 5.1 데이터 처리 파이프라인

python
def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
train_data, train_dataset = preprocessor.make_set_as_df(
os.path.join(data_path, 'train.csv'),
is_train=True
)
val_data, val_dataset = preprocessor.make_set_as_df(
os.path.join(data_path, 'val.csv'),
is_train=False
)
return train_dataset, val_dataset


### 5.2 PPO 학습 프로세스
python
def ppo_train_step(self, dataloader):
self.model.train()
for batch in tqdm(dataloader, desc="PPO Training"):
# 입력 데이터 처리
input_ids = batch['input_ids'].to(self.model.device)
attention_mask = batch['attention_mask'].to(self.model.device)
# 요약문 생성 및 보상 계산
generated_ids, new_log_probs = self.generate_with_log_probs(
input_ids,
attention_mask
)
# PPO 업데이트 수행
advantages = new_rewards - old_rewards
ratios = torch.exp(new_log_probs - old_log_probs)
# 손실 계산 및 모델 업데이트
policy_loss = -torch.min(surr1, surr2).mean()
loss = policy_loss - self.ppo_config['entropy_coef'] entropy
loss.backward()


## 6. 향후 개선 방향

### 6.1 성능 최적화
- 더 효율적인 메모리 사용을 위한 최적화 기법 도입
- 분산 학습 지원을 통한 학습 속도 향상

### 6.2 기능 확장
- 다양한 도메인의 텍스트에 대한 요약 성능 개선
- 다국어 지원 확장

### 6.3 모니터링 강화
- 학습 과정의 시각화 도구 개발
- 자동화된 성능 평가 시스템 구축

## 7. 결론

이번 개발 과정을 통해 대규모 언어 모델을 활용한 텍스트 요약 시스템의 구현과 최적화에 대한 다양한 경험을 얻을 수 있었습니다. 특히 메모리 관리, 텐서 연산 최적화, 그리고 효율적인 학습 프로세스 구축의 중요성을 실제 경험을 통해 깊이 이해할 수 있었습니다.

앞으로도 지속적인 개선과 최적화를 통해 더 나은 성능과 효율성을 갖춘 시스템으로 발전시켜 나갈 계획입니다.