import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from konlpy.tag import Hannanum

# csv
df = pd.read_csv('csv/sk하이닉스.csv')

# KoELECTRA 모델 로드
model_name = "monologg/kobert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_labels = 2  # 수정된 부분
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 제목과 내용을 합쳐서 content 열 생성
df['content'] = df['title']

# 감성 분석을 위한 전처리 함수
def preprocess(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs.to(device)
    return inputs

# 예측 함수
def predict(inputs):
    outputs = model(**inputs)
    logits = outputs.logits
    probs = logits.softmax(dim=-1)
    return probs[0].detach().cpu().numpy()

# content 열에 대해 예측 수행
df['sentiment'] = df['content'].apply(lambda x: predict(preprocess(x)))

# 감성 분석 결과를 기반으로 긍정, 부정, 중립 라벨링 함수 추가
def get_label(probs):
    if probs[0] <= 0.4:  # 0.4 이하면 부정
        return '부정'
    elif probs[0] >= 0.6:  # 0.6 이상이면 긍정
        return '긍정'

df['label'] = df['sentiment'].apply(get_label)


# 적자, 손실이라는 단어가 있으면 부정으로 분류하는 함수 추가 및 긍정, 중립 함수 추가
def check_negative(text):
    negative_words = ['적자', '손실', '빨간불', '위기', '몸살', '바닥', '부정', '뿔났다', '사망', '하락', '스톱', '부정적', '추락', '걱정', '부담', '침체', '부진']
    for word in negative_words:
        if word in text:
            return True
    return False

def check_positive(text):
    positive_words = ['혜택', '뛰어넘자', '탈출', '연장', '협약', '참여', '초록불', '위기탈출', '인센티브', '긍정', '해결', '상승', '긍정적', '지킨다', '지원사격', '순이익', '기대감', '인기', '활짝', '보조금']
    for word in positive_words:
        if word in text:
            return True
    return False

def check_neutral(text):
    neutral_words = ['반입', '진화', '전망', '모집', '목표', '도입', '공시', '연장', '방문', '활용', '강화', '협약', '의도', '합의', '연구', '점유율', '행사', '만났다', '세운다', '짓는다', '전파', '기술연구']
    for word in neutral_words:
        if word in text:
            return True
    return False

# label 열에 적용하여 부정으로 분류
df['label'] = df.apply(lambda x: '부정' if check_negative(x['content']) else x['label'], axis=1)


# 형태소 분석 함수 추가
hannanum = Hannanum()
def hannanum_tokenize(text):
    return hannanum.morphs(text)

df['hannanum_tokens'] = df['content'].apply(hannanum_tokenize)


# csv 파일로 저장
df.to_csv('csv/sk하이닉스_주가뉴스(KoBart season2).csv', index=False)

# 결과 확인
print(df[['content', 'sentiment', 'label']])