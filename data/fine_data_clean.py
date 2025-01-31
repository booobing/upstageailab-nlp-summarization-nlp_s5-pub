# 라이브러리 임포트
import pandas as pd
import os
from spellchecker import SpellChecker
from tqdm import tqdm

# 1. 오타 및 불필요한 공백 제거를 위한 함수 정의
def clean_text(text):
    # 기본적인 텍스트 정제만 수행
    # 연속된 공백을 하나로 통일
    text = ' '.join(text.split())
    # 문장 앞뒤 공백 제거
    text = text.strip()
    return text

# 2. 라벨 검증 및 정제를 위한 함수 정의
def verify_and_clean_labels(df):
    # summary 컬럼이 비어있는지 확인하고 제거
    df = df[df['summary'].notnull()]
    
    # 추가적인 라벨 검증 로직을 여기에 작성 (예: summary 길이 체크 등)
    # 예시: summary가 너무 짧은 경우 제거
    df = df[df['summary'].apply(lambda x: len(x.split()) > 5)]
    
    return df


# 3. 전체 데이터 전처리 함수 정의
def create_fine_data(input_path, output_path):
    # 데이터 불러오기
    df = pd.read_csv(input_path)
    
    # 텍스트 정제 - apply 대신 더 빠른 벡터화 연산 사용
    print("텍스트 정제 중...")
    
    # 'i:' 패턴 제거
    df['dialogue'] = df['dialogue'].str.replace(r'i', '', regex=True)
    df['summary'] = df['summary'].str.replace(r'i', '', regex=True)
    
    
    # 결과 저장
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    
    print(f"파인데이터가 '{output_path}'에 저장되었습니다.")

# 5. 메인 실행 함수
if __name__ == "__main__":
    DATA_PATH = "/home/data/"
    FINE_DATA_PATH = "/home/fine_data/"
    TRAIN_INPUT_FILE = os.path.join(FINE_DATA_PATH, 'fine_train_spellchecker.csv')
    FINE_TRAIN_OUTPUT_FILE = os.path.join(FINE_DATA_PATH, 'fine_train_spellchecker_cleaned.csv')
    
    create_fine_data(TRAIN_INPUT_FILE, FINE_TRAIN_OUTPUT_FILE)