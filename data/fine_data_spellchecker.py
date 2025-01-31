# 라이브러리 임포트
import pandas as pd
import os
from spellchecker import SpellChecker
from tqdm import tqdm

# 1. 오타 및 불필요한 공백 제거를 위한 함수 정의
def clean_text(text, spell_checker, common_words_cache=set()):
    """
    캐싱과 배치 처리를 활용한 빠른 텍스트 정제
    """
    # 기본 공백 정제
    text = ' '.join(text.split())
    
    # 이미 검사한 일반적인 단어들은 건너뛰기
    words = text.split()
    corrected_words = []
    
    # 배치로 모든 단어의 오타 여부 확인
    unknown_words = spell_checker.unknown(words)
    
    for word in words:
        if word in common_words_cache:
            corrected_words.append(word)
        elif word in unknown_words:
            corrected = spell_checker.correction(word)
            corrected_words.append(corrected if corrected else word)
            if len(common_words_cache) < 10000:  # 캐시 크기 제한
                common_words_cache.add(word)
        else:
            corrected_words.append(word)
            if len(common_words_cache) < 10000:  # 캐시 크기 제한
                common_words_cache.add(word)
    
    return ' '.join(corrected_words)

# 2. 라벨 검증 및 정제를 위한 함수 정의
def verify_and_clean_labels(df):
    # summary 컬럼이 비어있는지 확인하고 제거
    df = df[df['summary'].notnull()]
    
    # 추가적인 라벨 검증 로직을 여기에 작성 (예: summary 길이 체크 등)
    # 예시: summary가 너무 짧은 경우 제거
    df = df[df['summary'].apply(lambda x: len(x.split()) > 5)]
    
    return df

# 3. 특징 스키마 통일을 위한 함수 정의
def unify_schema(df):
    # 현재 데이터셋에 날짜 정보가 없으므로 예시로 다른 스키마 통일 작업을 진행
    # 예를 들어, 모든 텍스트을 소문자로 변환
    df['dialogue'] = df['dialogue'].str.lower()
    df['summary'] = df['summary'].str.lower()
    
    return df

# 4. 전체 데이터 전처리 함수 정의
def create_fine_data(input_path, output_path):
    # 데이터 불러오기
    df = pd.read_csv(input_path)
    
    # SpellChecker 인스턴스 한 번만 생성
    spell = SpellChecker()
    common_words_cache = set()
    
    # 오타 수정 및 공백 정제
    print("텍스트 정제 중...")
    # 배치 크기로 처리
    batch_size = 1000
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        df.loc[batch.index, 'dialogue'] = batch['dialogue'].apply(
            lambda x: clean_text(x, spell, common_words_cache))
        df.loc[batch.index, 'summary'] = batch['summary'].apply(
            lambda x: clean_text(x, spell, common_words_cache))
        
        # 진행률 표시
        print(f"처리 중: {min((i+batch_size)/len(df)*100, 100):.1f}%", end='\r')
    
    # 라벨 검증 및 정제
    df = verify_and_clean_labels(df)
    
    # 스키마 통일
    df = unify_schema(df)
    
    # 결과 저장
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    
    print(f"\n파인데이터가 '{output_path}'에 저장되었습니다.")

# 5. 메인 실행 함수
if __name__ == "__main__":
    DATA_PATH = "/home/data/"
    FINE_DATA_PATH = "/home/fine_data/"
    TRAIN_INPUT_FILE = os.path.join(DATA_PATH, 'train.csv')
    FINE_TRAIN_OUTPUT_FILE = os.path.join(FINE_DATA_PATH, 'fine_train_spellchecker.csv')
    
    create_fine_data(TRAIN_INPUT_FILE, FINE_TRAIN_OUTPUT_FILE)