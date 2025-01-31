# 4. 필요한 라이브러리 임포트
import pandas as pd
import os
import time
from tqdm import tqdm
from rouge import Rouge
from openai import OpenAI

UPSTAGE_API_KEY = "up_aiCe2KyaGnLmvgVLham0vcOEudzdc" # upstage.ai에서 발급받은 API KEY를 입력해주세요.

client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

# 데이터 경로를 지정해줍니다.
DATA_PATH = "/home/fine_data/"
RESULT_PATH = "./prediction/"

# train data의 구조와 내용을 확인합니다.
train_df = pd.read_csv(os.path.join(DATA_PATH,'fine_train_llm.csv'))

# validation data의 구조와 내용을 확인합니다.
val_df = pd.read_csv(os.path.join(DATA_PATH,'dev.csv'))

# 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.
rouge = Rouge()
def compute_metrics(pred, gold):
    results = rouge.get_scores(pred, gold, avg=True)
    result = {key: value["f"] for key, value in results.items()}
    return result

def validate_summary(summary):
    """
    주어진 요약문이 프롬프트의 요구사항을 충족하는지 확인하는 함수
    """
    # 1. 길이 제한 확인 (105자)
    if len(summary) > 105:
        return False, "Length exceeds 105 characters"
    
    # 2. 기본적인 형식 검증
    if not summary or summary.isspace():
        return False, "Empty or whitespace summary"
        
    # 3. Person1, Person2 형식 확인
    if "#Person" in summary and not all(x in summary for x in ["#Person1", "#Person2"]):
        return False, "Inconsistent person formatting"
    
    return True, "Valid summary"

def refine_with_solar(original_summary, dialogue):
    """
    Solar LLM을 사용하여 요약문을 개선하는 함수
    """
    system_prompt = (
        "You are an expert in dialogue summarization. Your task is to produce a concise and coherent summary of the given dialogue in korean.\n"
        "Ensure your summary:\n"
        "1. Maintains factual consistency (no fabricated details or omissions of critical events).\n"
        "2. Provides relevant background or context where necessary, so the reader understands the situation.\n"
        "3. Focuses on actions, decisions, and outcomes, explaining their significance.\n"
        "4. Use consistent #Person# formatting\n"
        "5. Avoids subjective or emotional language. Use direct, neutral descriptions of events in a formal tone.\n"
        "6. Is concise, avoiding unnecessary repetition or excessive detail.\n"
        "7. Preserves logical flow (the summary should read naturally and stay on-topic).\n"
        "8. Uses clear, natural language without random insertions or mistranslations.\n\n"
        "Your output will be evaluated using ROUGE metrics. Capture the essence of the original dialogue accurately and briefly."
    )
    
    user_prompt = (
        "Please follow the instructions below to summarize the given dialogue.\n\n"
        "Instructions (perform each step in order):\n"
        "STEP 1. Read the sample dialogue and its summary to understand the target style.\n"
        "STEP 2. Carefully read the new dialogue provided.\n"
        "STEP 3. Summarize the new dialogue while:\n"
        "   (1) Including all essential information about the core issues, conflicts, or decisions.\n"
        "   (2) Providing any necessary background or context so the actions and outcomes are clear.\n"
        "   (3) Maintaining an objective, formal tone without emotional or subjective judgments.\n"
        "   (4) Ensuring the entire summary does not exceed 105 characters (in total).\n"
        "STEP 4. Recheck the summary to confirm:\n"
        "   - It is factually consistent and logically flows.\n"
        "   - It omits trivial or repetitive details.\n"
        "   - It respects the 105-character limit.\n\n"
        "Remember, your summary will be evaluated using ROUGE metrics, so it must reflect the important points of the dialogue accurately.\n\n"
        f"Original dialogue:\n{dialogue}\n\n"
        f"Current summary:\n{original_summary}\n\n"
        "Please provide an improved summary following the rules above."
    )
    
    try:
        response = client.chat.completions.create(
            model="solar-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            top_p=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in refining summary: {e}")
        return original_summary
    
def process_output_with_validation(output_df):
    """
    출력 데이터프레임을 검증하고 필요한 경우 개선하는 함수
    """
    improved_summaries = []
    
    for idx, row in tqdm(output_df.iterrows(), total=len(output_df)):
        summary = row['summary']
        dialogue = row['dialogue']  # dialogue가 있는지 확인
        
        if not isinstance(dialogue, str):  # dialogue 타입 체크
            print(f"Warning: Invalid dialogue type at index {idx}")
            continue
            
        is_valid, message = validate_summary(summary)
        
        if not is_valid:
            print(f"\nInvalid summary found in {row['fname']}: {message}")
            print(f"Original summary: {summary}")
            print(f"Dialogue preview: {dialogue[:100]}...")  # dialogue 미리보기 출력
            
            # Solar LLM으로 요약 개선
            improved_summary = refine_with_solar(summary, dialogue)
            print(f"Improved summary: {improved_summary}")
            improved_summaries.append(improved_summary)
        else:
            improved_summaries.append(summary)
    
    # 개선된 요약으로 새로운 데이터프레임 생성
    improved_output = pd.DataFrame({
        'fname': output_df['fname'],
        'summary': improved_summaries
    })
    
    # 결과 저장
    improved_output.to_csv(os.path.join(RESULT_PATH, "output_improved.csv"), index=False)
    
    return improved_output

# 사용 예시
if __name__ == "__main__":
    # 파일 경로 확인
    print(f"Looking for files in: {DATA_PATH}")
    
    # 테스트 데이터와 출력 데이터 로드
    try:
        test_df = pd.read_csv(os.path.join('/home/data', 'test.csv'))
        print(f"Test data loaded: {len(test_df)} rows")
        print("Test data columns:", test_df.columns.tolist())
        
        output_df = pd.read_csv(os.path.join('/home/code/prediction', 'output_solar_pro_llm_improved2.csv'))
        print(f"Output data loaded: {len(output_df)} rows")
        print("Output data columns:", output_df.columns.tolist())
        
        # dialogue와 summary가 제대로 매칭되는지 확인
        merged_df = pd.DataFrame({
            'fname': output_df['fname'],
            'summary': output_df['summary'],
            'dialogue': test_df['dialogue']
        })
        
        # 데이터 샘플 출력으로 확인
        print("\nFirst row sample:")
        print("File name:", merged_df['fname'].iloc[0])
        print("Dialogue:", merged_df['dialogue'].iloc[0][:100], "...")  # 처음 100자만 출력
        print("Summary:", merged_df['summary'].iloc[0])
        
        # 검증 및 개선 프로세스 실행
        improved_output = process_output_with_validation(merged_df)
        
        # 결과 확인
        print("\nValidation and improvement process completed")
        print(f"Total processed items: {len(improved_output)}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required files - {e}")
    except Exception as e:
        print(f"Error during processing: {e}")