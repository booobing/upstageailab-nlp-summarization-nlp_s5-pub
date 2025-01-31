import pandas as pd

# 파일 경로 설정
file_path = './output.csv'

# CSV 파일 읽기
df = pd.read_csv(file_path, encoding='utf-8')

# summary 열의 글자 수 계산
df['summary_length'] = df['summary'].apply(len)

# 평균 글자 수 계산
average_length = df['summary_length'].mean()
print(f'Average length of summary: {average_length:.2f} characters')