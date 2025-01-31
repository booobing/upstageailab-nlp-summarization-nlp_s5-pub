# KoBART 기반 대화 요약 모델 개발 및 개선 과정

## 1. 프로젝트 개요

### 목적
- KoBART를 활용한 대화 요약 모델 개발
- 모델의 성능 평가 및 개선
- 평가 메트릭으로 ROUGE 스코어 사용

### 사용 기술
- 기본 모델: digit82/kobart-summarization
- 프레임워크: PyTorch, Transformers
- 평가 메트릭: ROUGE Score

## 2. 주요 문제점 및 해결 과정

### 2.1. 평가 메트릭 키 불일치 문제

#### 문제 상황
- `KeyError: 'rouge1'` 오류 발생
- 평가 메트릭 키 이름이 일치하지 않음
- 로그에서는 `'eval_rouge-1'`, `'eval_rouge-2'`, `'eval_rouge-l'`로 표시됨

#### 해결 방안
- 메트릭 키 이름을 통일
  - `'rouge1'` → `'eval_rouge-1'`
  - `'rouge2'` → `'eval_rouge-2'`
  - `'rougel'` → `'eval_rouge-l'`
- `compute_metrics` 함수 수정하여 일관된 키 네이밍 적용

### 2.2. 모델 체크포인트 로딩 문제

#### 문제 상황
- 체크포인트 로드 시 누락된 키 경고 발생
  ```
  missing keys in the checkpoint model loaded: 
  ['model.encoder.embed_tokens.weight', 
   'model.decoder.embed_tokens.weight', 
   'lm_head.weight']
  ```

#### 해결 방안
1. 체크포인트 저장 시 전체 모델 저장하도록 수정
2. 토크나이저와 모델의 특수 토큰 일치시킴
3. `resize_token_embeddings` 함수 호출하여 임베딩 크기 조정

### 2.3. GPU 메모리 관리 문제

#### 문제 상황
- GPU 메모리 부족으로 인한 OOM(Out of Memory) 오류
- 큰 배치 사이즈로 인한 메모리 사용량 증가

#### 해결 방안
1. 배치 사이즈 조정
   - 학습: 50 → 16
   - 평가: 32 → 16
2. Gradient Accumulation Steps 설정
3. GPU 캐시 정리 코드 추가
   ```python
   torch.cuda.empty_cache()
   ```

### 2.4. 코드 중복 문제

#### 문제 상황
- `load_tokenizer_and_model`과 `load_tokenizer_and_model_for_test` 함수의 중복
- 유사한 기능을 하는 두 개의 함수 존재

#### 해결 방안
- 두 함수를 하나로 통합
- `is_train` 파라미터를 추가하여 용도에 따라 구분
- 체크포인트 경로를 동적으로 설정

## 3. 성능 평가 개선

### 3.1. compute_metrics 함수 개선

#### 이전 버전의 문제점
- 예측값 처리 방식이 불완전
- 특수 토큰 처리가 미흡
- 메트릭 키 네이밍 불일치

#### 개선된 버전
python
def compute_metrics(config, tokenizer, pred):
rouge = Rouge()
predictions = pred.predictions
labels = pred.label_ids
if isinstance(predictions, tuple):
predictions = predictions[0]
predictions = np.argmax(predictions, axis=-1)
predictions = predictions.astype(np.int64)
labels = labels.astype(np.int64)
labels[labels == -100] = tokenizer.pad_token_id
decoded_preds = tokenizer.batch_decode(
predictions.tolist(),
skip_special_tokens=True,
clean_up_tokenization_spaces=True
)
decoded_labels = tokenizer.batch_decode(
labels.tolist(),
skip_special_tokens=True,
clean_up_tokenization_spaces=True
)
remove_tokens = config['inference']['remove_tokens']
for token in remove_tokens:
decoded_preds = [sentence.replace(token, " ") for sentence in decoded_preds]
decoded_labels = [sentence.replace(token, " ") for sentence in decoded_labels]
results = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
result = {
'eval_rouge-1': results['rouge-1']['f'],
'eval_rouge-2': results['rouge-2']['f'],
'eval_rouge-l': results['rouge-l']['f']
}
return result


### 3.2. 모델 설정 최적화

#### 주요 설정 변경사항

python
config_data = {
"training": {
"per_device_train_batch_size": 16,
"per_device_eval_batch_size": 16,
"gradient_accumulation_steps": 2,
"warmup_ratio": 0.1,
"learning_rate": 1e-5,
"num_train_epochs": 20,
"early_stopping_patience": 3,
"early_stopping_threshold": 0.001,
}
}


## 4. 추가 개선사항

### 4.1. 경고 메시지 처리
- torch.utils.checkpoint 관련 경고 해결
- gradient_checkpointing 설정 명시적 지정

### 4.2. 데이터 전처리 개선
- 특수 토큰 처리 방식 통일
- skip_special_tokens 옵션 일관되게 적용

### 4.3. 환경 설정 표준화
- 라이브러리 버전 통일
- 랜덤 시드 고정

## 5. 향후 개선 방향

### 5.1. 성능 최적화
- 배치 사이즈와 학습률 추가 조정
- 모델 아키텍처 최적화 검토

### 5.2. 코드 품질 개선
- 중복 코드 추가 제거
- 에러 처리 강화
- 로깅 시스템 개선

### 5.3. 평가 메트릭 확장
- ROUGE 외 추가 평가 지표 도입 검토
- 사람 평가와의 상관관계 분석

## 6. 결론

이 프로젝트를 통해 KoBART 기반 대화 요약 모델의 개발과 평가 과정에서 발생하는 다양한 문제점들을 확인하고 해결했습니다. 특히 평가 메트릭 처리, GPU 메모리 관리, 코드 중복 제거 등의 측면에서 큰 개선을 이루었습니다. 향후에도 지속적인 성능 최적화와 코드 품질 개선을 통해 모델의 완성도를 높여나갈 계획입니다.