import pandas as pd
import os
import time
from tqdm import tqdm
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.
from openai import OpenAI # openai==1.2.0
from spellchecker import SpellChecker

# Solar Chat API Client 생성
UPSTAGE_API_KEY = "up_aiCe2KyaGnLmvgVLham0vcOEudzdc" 
client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

# 1. 오타 및 불필요한 공백 제거를 위한 함수 정의
def clean_text_with_solar(text):
    # Solar API를 사용하여 텍스트 정제
    response = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=[
            {
                "role": "system", 
                "content": "당신은 텍스트 정제 전문가입니다. 주어진 텍스트의 오타를 수정하고 불필요한 공백을 제거해주세요."
            },
            {
                "role": "user",
                "content": f"다음 텍스트를 정제해주세요:\n{text}"
            }
        ],
        temperature=0.2,
        top_p=0.3,
    )
    
    return response.choices[0].message.content

# 2. 라벨 검증 및 정제를 위한 함수 정의
def verify_and_clean_labels_with_solar(df):
    # summary 컬럼이 비어있는지 확인하고 제거
    df = df[df['summary'].notnull()]
    
    # Solar API를 사용하여 라벨 검증
    verified_summaries = []
    start_time = time.time()
    
    for idx, summary in enumerate(tqdm(df['summary'], desc="라벨 검증 중")):
        response = client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 텍스트 검증 전문가입니다. 주어진 요약문이 적절한지 검증해주세요."
                },
                {
                    "role": "user",
                    "content": f"다음 요약문이 적절한지 검증해주세요. 5단어 이상이어야 하며, 문법적으로 올바르고 의미가 명확해야 합니다:\n{summary}"
                }
            ],
            temperature=0.2,
            top_p=0.3,
        )
        verified_summaries.append(response.choices[0].message.content)
        
        # Rate limit 방지를 위해 1분 동안 최대 100개의 요청을 보내도록 합니다.
        if (idx + 1) % 100 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time + 5
                print(f"Elapsed time: {elapsed_time:.2f} sec")
                print(f"Waiting for {wait_time} sec")
                time.sleep(wait_time)
            
            start_time = time.time()
    
    df['summary'] = verified_summaries
    return df

# 3. 특징 스키마 통일을 위한 함수 정의
def unify_schema_with_solar(df):
    # Solar API를 사용하여 스키마 통일
    unified_dialogues = []
    unified_summaries = []
    start_time = time.time()
    
    for idx, (dialogue, summary) in enumerate(tqdm(zip(df['dialogue'], df['summary']), desc="스키마 통일 중", total=len(df))):
        # 대화문 통일
        dialogue_response = client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 텍스트 형식 통일 전문가입니다. 주어진 텍스트의 형식을 통일해주세요."
                },
                {
                    "role": "user",
                    "content": f"다음 대화문의 형식을 통일해주세요:\n{dialogue}"
                }
            ],
            temperature=0.2,
            top_p=0.3,
        )
        unified_dialogues.append(dialogue_response.choices[0].message.content)
        
        # 요약문 통일
        summary_response = client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 텍스트 형식 통일 전문가입니다. 주어진 텍스트의 형식을 통일해주세요."
                },
                {
                    "role": "user",
                    "content": f"다음 요약문의 형식을 통일해주세요:\n{summary}"
                }
            ],
            temperature=0.2,
            top_p=0.3,
        )
        unified_summaries.append(summary_response.choices[0].message.content)
        
        # Rate limit 방지를 위해 1분 동안 최대 100개의 요청을 보내도록 합니다.
        if (idx + 1) % 400 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time + 5
                print(f"Elapsed time: {elapsed_time:.2f} sec")
                print(f"Waiting for {wait_time} sec")
                time.sleep(wait_time)
            
            start_time = time.time()
    
    df['dialogue'] = unified_dialogues
    df['summary'] = unified_summaries
    return df

# 4. 전체 데이터 전처리 함수 정의
def create_fine_data(input_path, output_path):
    # 데이터 불러오기
    df = pd.read_csv(input_path)
    
    # 오타 수정 및 공백 정제
    tqdm.pandas(desc="텍스트 정제 중")
    df['dialogue'] = df['dialogue'].progress_apply(clean_text_with_solar)
    df['summary'] = df['summary'].progress_apply(clean_text_with_solar)
    
    # 라벨 검증 및 정제
    df = verify_and_clean_labels_with_solar(df)
    
    # 스키마 통일
    df = unify_schema_with_solar(df)
    
    # 결과 저장
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    
    print(f"파인데이터가 '{output_path}'에 저장되었습니다.")

# 5. 메인 실행 함수
if __name__ == "__main__":
    DATA_PATH = "/home/data/"
    FINE_DATA_PATH = "/home/fine_data/"
    TRAIN_INPUT_FILE = os.path.join(DATA_PATH, 'train.csv')
    FINE_TRAIN_OUTPUT_FILE = os.path.join(FINE_DATA_PATH, 'fine_train.csv')
    
    create_fine_data(TRAIN_INPUT_FILE, FINE_TRAIN_OUTPUT_FILE) 