import pandas as pd
import torch
import sentencepiece as spm
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from konlpy.tag import Hannanum
import random
import numpy as np

# csv
df = pd.read_csv('csv/아이돌.csv')

# 랜덤 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()  # 랜덤 시드 설정

# KoELECTRA 모델 로드 (분류 모델로 변경)
model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = ElectraTokenizer.from_pretrained(model_name)
num_labels = 2  # 긍정, 부정, 중립 3개의 라벨
model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 제목과 내용을 합쳐서 content 열 생성
df['content'] = df['title']

# 감성 분석을 위한 전처리 함수
def preprocess(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    # inputs를 GPU로 이동
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs

# 예측 함수
def predict(inputs):
    try:
        outputs = model(**inputs)
        logits = outputs.logits
        probs = logits.softmax(dim=-1)
        return probs[0].detach().cpu().numpy()
    except Exception as e:
        print(f"예측 오류: {e}")
        return [0.0, 0.0, 0.0]  # 오류 시 기본값 반환

# content 열에 대해 예측 수행
df['sentiment'] = df['content'].apply(lambda x: predict(preprocess(x)))

# 감성 분석 결과를 기반으로 긍정, 부정, 중립 라벨링 함수
def get_label(probs):
    # 확률을 소수점 둘째 자리까지 반올림하여 비교
    probs = [round(p, 2) for p in probs]
    if probs[0] <= 0.49:  # 0.50 미만이면 부정
        return '부정'
    elif probs[0] >= 0.51:  # 0.50 초과이면 긍정
        return '긍정'
    else:  # 0.50이면 중립
        return '중립'

df['label'] = df['sentiment'].apply(get_label)

# 적자, 손실이라는 단어가 있으면 부정으로 분류하는 함수 추가 및 긍정, 중립 함수 추가
def check_negative(text):
    negative_words = ['가스라이팅', '논란', '폄하', '면 뭐해?', '애물단지', '적자', '손실', '빨간불', '위기', '몸살', '바닥', '부정', '뿔났다', '사망', '하락', '스톱', '부정적', '추락', '걱정', '부담', '침체', '부진', '높은 벽', '최저가보장제 없앴다', '매도', '갑질', '방치']
    return any(word in text for word in negative_words)

def check_positive(text):
    positive_words = ['가능성', '개선', '응원', '혜택', '뛰어넘자', '탈출', '연장', '협약', '참여', '초록불', '위기탈출', '인센티브', '긍정', '해결', '상승', '긍정적', '지킨다', '지원사격', '순이익', '기대감', '인기', '활짝', '보조금']
    return any(word in text for word in positive_words)

def check_neutral(text):
    neutral_words = ['막판 포기', '논의 중단', '왜?', '매도', '포럼', '불참', '최후통첩', '긴급 면담']
    return any(word in text for word in neutral_words)

# 긍정적인 단어가 있는 경우 긍정으로 강제 라벨링
df['label'] = df.apply(lambda x: '긍정' if check_positive(x['content']) else x['label'], axis=1)

# label 열에 적용하여 부정으로 분류
df['label'] = df.apply(lambda x: '부정' if check_negative(x['content']) else x['label'], axis=1)

# label 열에 적용하여 중립으로 분류
df['label'] = df.apply(lambda x: '중립' if check_neutral(x['content']) else x['label'], axis=1)

# Hannanum 형태소 분석기 사용
hannanum = Hannanum()
def hannanum_tokenize(text):
    return hannanum.morphs(text)

df['hannanum_tokens'] = df['content'].apply(hannanum_tokenize)

# 최종 확인
print(df[['date', 'link', 'content', 'sentiment', 'label', 'hannanum_tokens']])

# csv 파일로 저장
df.to_csv('csv/아이돌_뉴스.csv', index=False)

# 결과 확인
print(df[['date', 'link','content', 'sentiment', 'label']])