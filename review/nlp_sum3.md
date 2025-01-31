# 데이터 전처리 개발 과정 및 시행착오 정리

## 1. 초기 개발 목표
- train 데이터를 파인 데이터로 변환하는 기능 개발
- 파인 데이터의 조건:
  - 정확한 라벨링
  - 오타, 누락 최소화
  - 특징(Feature) 스키마 통일
  - 라벨링 기준 통일
  - 플랫폼/도메인별 통합 기준

## 2. 개발 과정

### 2.1 첫 번째 접근: Solar LLM API 활용
- Solar LLM API를 사용하여 텍스트 정제 및 검증 시도
- 구현 기능:
  - 텍스트 정제 (clean_text_with_solar)
  - 라벨 검증 (verify_and_clean_labels_with_solar)
  - 스키마 통일 (unify_schema_with_solar)

### 2.2 두 번째 접근: Rate Limit 관리
- API 호출 제한 문제 발생
- 해결책:
  - 1분당 100개 요청으로 제한
  - 대기 시간 추가 (60초 + 5초 여유)
- 코드 예시:
python
if (idx + 1) % 100 == 0:
end_time = time.time()
elapsed_time = end_time - start_time
if elapsed_time < 60:
wait_time = 60 - elapsed_time + 5
time.sleep(wait_time)
start_time = time.time()


### 2.3 세 번째 접근: 병렬 처리 구현
- 처리 속도 개선을 위한 ThreadPoolExecutor 도입
- 주요 변경사항:
  - 동시 실행 스레드 수 조절 (max_workers)
  - 대화문과 요약문 동시 처리
  - 결과 수집 및 정렬

### 2.4 네 번째 접근: 오류 처리 개선
- API 호출 실패 시 프로세스 중단 문제 발생
- 해결책:
  - 실패한 항목 추적 기능 추가
  - 원본 텍스트 보존
  - 오류 발생 지점 로깅

### 2.5 최종 접근: 로컬 처리로 전환
- API 호출 방식의 한계 발견:
  - 처리 속도가 느림
  - API 비용 발생
  - Rate Limit 제약
- 해결책:
  - SpellChecker 라이브러리 활용
  - 로컬 처리로 전환하되 병렬 처리 유지

## 3. 시행착오 및 교훈

### 3.1 API 관련 문제
- Rate Limit 관리의 중요성
- API 호출 실패 처리의 필요성
- 비용과 속도의 trade-off 고려

### 3.2 병렬 처리 관련
- ThreadPoolExecutor 활용의 이점
- 적절한 max_workers 값 설정의 중요성
- 결과 수집 및 정렬 로직 필요성

### 3.3 오류 처리 관련
- 초기 단계부터 오류 처리 구현의 중요성
- 데이터 손실 방지를 위한 안전장치 필요
- 실패 지점 추적의 유용성

## 4. 최종 결론
- 로컬 처리 + 병렬 처리 조합이 가장 효율적
- 초기 설계 시 오류 처리 고려 필요
- 실제 사용 환경에 맞는 접근 방식 선택 중요

## 5. 향후 개선 방향
- 메모리 사용량 최적화
- 진행 상황 모니터링 개선
- 재처리 기능 추가
- 로깅 시스템 강화