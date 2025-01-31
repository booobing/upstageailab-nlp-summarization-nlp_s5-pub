# PPO 학습 과정 개선 및 문제 해결 기록

## 1. 초기 문제점
- TensorBoard 설치 누락으로 인한 오류 발생
- gradient_clipping 대신 max_grad_norm 파라미터 사용 필요
- 패딩 토큰으로 인한 NaN 값 발생 문제

## 2. Best 모델 저장 관련 이슈
### 초기 상태
- Best 모델이 자동으로 저장되지 않는 문제 발생

### 해결 과정
1. Seq2SeqTrainingArguments 설정 수정
   - evaluation_strategy="steps" 설정
   - save_strategy="steps" 설정
   - metric_for_best_model 설정 추가
   - load_best_model_at_end=True 설정

2. ROUGE 점수 평균 기준 도입
   - rouge1, rouge2, rougeL의 평균값을 기준으로 변경
   - compute_metrics 함수에 rouge_avg 메트릭 추가

## 3. Gradient Checkpoint 관련 문제
### 발생한 문제들
1. "input_ids cannot be None" 오류
2. requires_grad=True 설정 관련 경고
3. 텐서 타입 불일치 문제

### 시도한 해결 방법들
1. 임베딩 사용 시도
   - input_ids를 임베딩으로 변환
   - requires_grad=True 설정
   - BART 모델 특성상 실패

2. input_ids 직접 사용
   - BART 모델이 input_ids를 필수로 요구
   - 임베딩 레이어 파라미터에 requires_grad=True 설정
   - attention_mask를 float 타입으로 변환

## 4. 최종 개선사항
1. 모델 설정
   - gradient_checkpointing_enable() 활성화
   - use_cache=False 설정

2. 텐서 처리
   - attention_mask를 float 타입으로 변환 후 requires_grad=True 설정
   - 임베딩 레이어 파라미터에 requires_grad=True 설정
   - input_ids는 정수 텐서로 유지

3. 학습 프로세스
   - PPO 업데이트 로직 개선
   - 그래디언트 계산 경로 명확화
   - 중복 코드 제거 및 통합

## 5. 교훈 및 시사점
1. BART 모델의 특성 이해 중요성
   - input_ids가 필수 파라미터
   - 임베딩 레이어 처리 방식

2. 그래디언트 계산 관련
   - 텐서 타입과 requires_grad 설정의 중요성
   - gradient checkpoint 사용 시 주의사항

3. 모델 저장 전략
   - 평가 메트릭 선정의 중요성
   - 자동 저장 메커니즘 활용

4. 코드 구조화
   - 중복 제거의 중요성
   - 명확한 책임 분리 