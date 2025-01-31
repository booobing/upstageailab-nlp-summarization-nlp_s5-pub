import pandas as pd
import os
import time
from tqdm import tqdm
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.
from openai import OpenAI # openai==1.2.0
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def create_fine_data(input_path, output_path, max_workers=10):
    # 데이터 불러오기
    df = pd.read_csv(input_path)
    
    # ThreadPoolExecutor를 사용하여 병렬 처리
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 오타 수정 및 공백 정제
        print("텍스트 정제 중...")
        dialogue_futures = {executor.submit(clean_text_with_solar, text): idx 
                          for idx, text in enumerate(df['dialogue'])}
        summary_futures = {executor.submit(clean_text_with_solar, text): idx 
                         for idx, text in enumerate(df['summary'])}
        
        # dialogue 결과 수집
        dialogue_results = {}
        failed_dialogues = []
        for future in tqdm(as_completed(dialogue_futures), total=len(dialogue_futures), desc="대화문 정제 중"):
            idx = dialogue_futures[future]
            try:
                dialogue_results[idx] = future.result()
            except Exception as e:
                print(f"대화문 처리 중 오류 발생 (인덱스 {idx}): {e}")
                failed_dialogues.append(idx)
                dialogue_results[idx] = df['dialogue'].iloc[idx]
        
        # summary 결과 수집
        summary_results = {}
        failed_summaries = []
        for future in tqdm(as_completed(summary_futures), total=len(summary_futures), desc="요약문 정제 중"):
            idx = summary_futures[future]
            try:
                summary_results[idx] = future.result()
            except Exception as e:
                print(f"요약문 처리 중 오류 발생 (인덱스 {idx}): {e}")
                failed_summaries.append(idx)
                summary_results[idx] = df['summary'].iloc[idx]
        
        # 결과를 DataFrame에 적용
        df['dialogue'] = [dialogue_results[i] for i in range(len(df))]
        df['summary'] = [summary_results[i] for i in range(len(df))]
    
    # 라벨 검증 (5단어 이상인지 확인)
    df = df[df['summary'].apply(lambda x: len(str(x).split()) >= 5)]
    
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
    
    create_fine_data(TRAIN_INPUT_FILE, FINE_TRAIN_OUTPUT_FILE, max_workers=10)