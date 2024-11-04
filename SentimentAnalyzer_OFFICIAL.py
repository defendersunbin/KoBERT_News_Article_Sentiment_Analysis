import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from konlpy.tag import Hannanum
import random
import numpy as np

# csv 파일 읽기
df = pd.read_csv('csv/아이돌.csv')

# 랜덤 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()  # 랜덤 시드 설정

# KcBERT 모델 로드
model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 감성 분석을 위한 전처리 함수
def preprocess(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs

# 예측 함수
def predict(inputs):
    try:
        with torch.no_grad():  # 그래디언트 계산 비활성화
            outputs = model(**inputs)
            logits = outputs.logits
            probs = logits.softmax(dim=-1)
            # 확률이 가장 높은 레이블의 인덱스 반환
            return np.argmax(probs[0].detach().cpu().numpy())
    except Exception as e:
        print(f"예측 오류: {e}")
        return -1  # 오류 시 기본값 반환

# content 열에 대해 예측 수행
df['evaluation'] = df['content'].apply(lambda x: predict(preprocess(x)))

# 평가 결과를 긍정, 부정, 중립으로 변환하는 함수
def label_evaluation(index):
    if index == 0:
        return '부정'
    elif index == 1:
        return '긍정'
    # elif index == 2:
    #     return '긍정'
    else:
        return '알 수 없음'

df['result'] = df['evaluation'].apply(label_evaluation)

# Hannanum 형태소 분석기 사용
hannanum = Hannanum()
def hannanum_tokenize(text):
    return hannanum.morphs(text)

df['hannanum_tokens'] = df['content'].apply(hannanum_tokenize)

# 최종 결과 확인
print(df[['content', 'result', 'hannanum_tokens']])

# csv 파일로 저장
df.to_csv('csv/아이돌_뉴스_1.csv', index=False)

# 결과 확인
print(df[['content', 'result']])