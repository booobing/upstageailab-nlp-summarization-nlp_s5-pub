# 대화 요약 모델 개발 과정 정리

## 1. 개발 목표
- BART 기반의 대화 요약 모델 개발
- wandb를 활용한 모델 학습 과정 모니터링 구현

## 2. 개발 과정

### 2.1 초기 설정
- baseline 코드를 기반으로 wandb 설정을 test_wandb.py에 구현
- 주요 구성:
  - config 설정 (wandb 섹션 추가)
  - trainer 설정
  - 모델 체크포인트 관리

### 2.2 발생한 문제점과 해결 과정

#### 1) wandb 모델 로깅 설정 문제
- 문제: WANDB_LOG_MODEL 환경변수 설정 오류
- 해결:
  ```python
  # 변경 전
  os.environ["WANDB_LOG_MODEL"]="true"
  
  # 변경 후
  os.environ["WANDB_LOG_MODEL"]="end"  # 또는 "checkpoint"
  ```
- 개선 효과:
  - "end": 학습 완료 후 최종 모델만 저장
  - "checkpoint": 각 체크포인트마다 모델 저장 가능

#### 2) 모델 체크포인트 경로 문제
- 문제: 잘못된 체크포인트 경로 설정
- 해결:
  ```python
  # 변경 전
  "ckt_path": "./model_output/checkpoint-epoch"
  
  # 변경 후
  "ckt_path": "./model_output/checkpoint-2490"
  ```
- 개선 효과:
  - 실제 존재하는 체크포인트 경로 지정
  - transformers 4.35.2 버전 호환성 확보
  - vocab_size 30008 지원

#### 3) 의존성 패키지 문제
- 문제: paramiko 모듈 미설치
- 해결: 필요한 패키지 추가 설치
  ```python
  !pip install wandb
  !pip install paramiko
  !pip install scp
  ```
- 개선 효과:
  - wandb 원격 로깅 기능 정상 작동
  - 안정적인 모델 학습 환경 구축

## 3. 최종 구현 기능
1. wandb를 통한 학습 과정 모니터링
2. 모델 체크포인트 자동 저장
3. 학습 지표 실시간 시각화
4. 안정적인 모델 배포 환경 구축

## 4. 향후 개선 방향
1. 체크포인트 관리 전략 최적화
2. wandb 로깅 커스터마이징
3. 모델 성능 지표 다각화

## 5. 교훈
1. 환경 변수 설정의 중요성
2. 의존성 패키지 관리의 필요성
3. 체계적인 모델 저장 전략 수립의 중요성

## 6. SFTP 파일 전송 시스템 개발

### 6.1 개발 목표
- 원격 서버와 로컬 PC 간의 안정적인 파일 전송 시스템 구축
- 대용량 데이터 전송을 위한 효율적인 전송 방식 구현

### 6.2 발생한 문제점과 해결 과정

#### 1) 경로 설정 문제
- 문제: Windows와 Linux 간 경로 표기 방식 차이로 인한 오류
- 해결:
  ```python
  # 변경 전
  pem_path = "C:\\Users\\user\\Downloads\\nlp_stage.pem"
  
  # 변경 후
  pem_path = "/home/nlp_stage.pem"  # Linux 경로 형식으로 통일
  ```
- 개선 효과:
  - 크로스 플랫폼 호환성 확보
  - 경로 관련 오류 해결

#### 2) SFTP 연결 안정성 문제
- 문제: 연결 실패 시에도 파일 전송 시도
- 해결:
  - 연결 상태 확인 로직 추가
  - try-except-finally 구문을 통한 예외 처리 강화
- 개선 효과:
  - 안정적인 파일 전송 보장
  - 오류 발생 시 명확한 피드백 제공

#### 3) 디렉토리 구조 처리
- 문제: 재귀적 디렉토리 전송 시 구조 유지 문제
- 해결:
  - 재귀 함수를 통한 디렉토리 구조 보존
  - 자동 디렉토리 생성 기능 구현
- 개선 효과:
  - 복잡한 디렉토리 구조도 온전히 전송 가능
  - 누락되는 파일 없이 완벽한 전송 보장

### 6.3 주요 구현 기능
1. 원격 서버에서 로컬로의 파일 다운로드
2. 재귀적 디렉토리 구조 처리
3. 진행 상황 실시간 모니터링
4. 안정적인 오류 처리 및 복구

### 6.4 향후 개선 방향
1. 전송 속도 최적화
2. 진행률 표시 기능 추가
3. 파일 무결성 검증 기능 구현
4. 동시 전송 처리 기능 추가

### 6.5 교훈
1. 크로스 플랫폼 호환성 고려의 중요성
2. 안정적인 예외 처리의 필요성
3. 사용자 피드백의 중요성

## 7. OpenAI API 연동 시스템 개발

### 7.1 개발 목표
- OpenAI API를 활용한 텍스트 처리 시스템 구축
- 안정적인 라이브러리 의존성 관리

### 7.2 발생한 문제점과 해결 과정

#### 1) 라이브러리 호환성 문제
- 문제: httpx와 OpenAI 패키지 간 버전 충돌
- 에러 내용: `ImportError: cannot import name 'BaseTransport' from 'httpx'`
- 해결:
  ```python
  !pip install --upgrade httpx
  !pip install --upgrade openai
  ```
- 개선 효과:
  - OpenAI 패키지와 httpx 라이브러리 간 호환성 확보
  - 안정적인 API 호출 환경 구축

#### 2) 필수 라이브러리 설정
- 구현 내용:
  ```python
  !pip install pandas
  !pip install googletrans==3.1.0a0
  ```
- 주요 기능:
  - pandas: 데이터프레임 처리
  - googletrans: 번역 기능 지원
  - tqdm: 진행률 표시
  - rouge: 모델 성능 평가

### 7.3 주요 구현 기능
1. OpenAI API 연동
2. 데이터프레임 기반 텍스트 처리
3. 다국어 번역 지원
4. 모델 성능 평가 시스템

### 7.4 향후 개선 방향
1. 라이브러리 버전 관리 자동화
2. 의존성 충돌 모니터링 시스템 구축
3. API 호출 최적화
4. 에러 처리 강화

### 7.5 교훈
1. 라이브러리 버전 호환성 확인의 중요성
2. 단계적 의존성 해결의 필요성
3. 체계적인 에러 로깅의 중요성 

## 8. Solar LLM을 활용한 요약문 개선 시스템 개발

### 8.1 개발 목표
- Solar LLM을 활용한 요약문 품질 개선
- 대량의 요약문 자동 처리 시스템 구축
- 실시간 진행 상황 모니터링 구현

### 8.2 발생한 문제점과 해결 과정

#### 1) OpenAI 라이브러리 호환성 문제
- 문제: httpx와 OpenAI 패키지 간 버전 충돌
- 에러 내용: `ImportError: cannot import name 'BaseTransport' from 'httpx'`
- 해결:
  ```python
  !pip uninstall -y httpx openai
  !pip install httpx==0.24.1
  !pip install openai==1.2.0
  ```
- 개선 효과:
  - OpenAI 패키지와 httpx 라이브러리 간 호환성 확보
  - 안정적인 API 호출 환경 구축

#### 2) 요약문 검증 및 개선 프로세스
- 문제: 요약문 품질 검증과 개선 과정의 비효율성
- 해결:
  - 자동화된 검증 시스템 구축
  - Solar LLM 프롬프트 최적화
  - 중간 결과 저장 기능 구현
- 개선 효과:
  - 대량의 요약문 효율적 처리
  - 일관된 품질 유지
  - 진행 상황 실시간 모니터링

#### 3) 데이터 처리 안정성
- 문제: dialogue와 summary 매칭 오류
- 해결:
  - 데이터프레임 병합 로직 개선
  - 예외 처리 강화
  - 진행 상황 로깅 추가
- 개선 효과:
  - 안정적인 데이터 처리
  - 오류 발생 시 즉각적인 파악 가능
  - 처리 과정 투명성 확보

### 8.3 주요 구현 기능
1. Solar LLM 기반 요약문 개선
   - 105자 제한 준수
   - 일관된 형식 유지
   - 핵심 내용 보존

2. 자동화된 처리 시스템
   - 대량 요약문 일괄 처리
   - 중간 결과 자동 저장
   - 진행 상황 모니터링

3. 데이터 관리 시스템
   - 파일 단위 처리
   - 에러 로깅
   - 결과 백업

### 8.4 향후 개선 방향
1. 프롬프트 엔지니어링 최적화
2. 처리 속도 개선
3. 모니터링 시스템 고도화
4. 에러 복구 시스템 강화

### 8.5 교훈
1. 라이브러리 버전 관리의 중요성
2. 단계적 시스템 개선의 필요성
3. 안정적인 데이터 처리의 중요성
4. 실시간 모니터링의 가치

## 9. 번역 시스템 개발 및 개선 과정

### 9.1 개발 목표
- 대화 데이터의 자동 번역 시스템 구축
- 번역 진행 상황 실시간 모니터링 구현
- 안정적인 번역 처리 환경 구축

### 9.2 발생한 문제점과 해결 과정

#### 1) googletrans 라이브러리 버전 문제
- 문제: httpcore.SyncHTTPTransport 관련 오류 발생
- 해결:
  ```python
  # 변경 전
  !pip install googletrans==3.1.0a0
  
  # 변경 후
  !pip uninstall googletrans -y
  !pip install googletrans==4.0.0-rc1
  ```
- 개선 효과:
  - 안정적인 번역 API 호출
  - 호환성 문제 해결

#### 2) 번역 진행 상황 모니터링
- 문제: 번역 진행 상황 파악 불가
- 해결:
  - tqdm 라이브러리 도입
  - 로깅 시스템 구축
  ```python
  from tqdm import tqdm
  import logging
  
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s'
  )
  ```
- 개선 효과:
  - 실시간 진행률 확인 가능
  - 에러 발생 시 즉각적인 파악
  - 번역 프로세스 투명성 확보

#### 3) 번역 오류 처리
- 문제: 번역 실패 시 전체 프로세스 중단
- 해결:
  - try-except 구문을 통한 예외 처리
  - 실패 시 원본 텍스트 유지 로직 추가
  ```python
  def back_translate(text, source_lang='ko', target_lang='en'):
      try:
          translated = translator.translate(text, src=source_lang, dest=target_lang).text
          back_translated = translator.translate(translated, src=target_lang, dest=source_lang).text
          return back_translated
      except Exception as e:
          logging.error(f"Translation error: {str(e)}")
          return text
  ```
- 개선 효과:
  - 안정적인 번역 프로세스 유지
  - 오류 발생 시에도 중단 없이 진행
  - 실패 케이스 추적 가능

### 9.3 주요 구현 기능
1. 자동화된 번역 시스템
   - 한영 번역
   - 역번역(back translation)
   - 배치 처리

2. 모니터링 시스템
   - 진행률 표시
   - 에러 로깅
   - 실시간 상태 확인

3. 오류 처리 시스템
   - 예외 상황 자동 처리
   - 로그 기록
   - 복구 메커니즘

### 9.4 향후 개선 방향
1. 번역 품질 개선
2. 처리 속도 최적화
3. 병렬 처리 도입 검토
4. 모니터링 시스템 고도화

### 9.5 교훈
1. 라이브러리 버전 관리의 중요성
2. 진행 상황 모니터링의 필요성
3. 안정적인 오류 처리의 가치
4. 단계적 시스템 개선의 효과