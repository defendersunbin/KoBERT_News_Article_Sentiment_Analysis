import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from konlpy.tag import Hannanum
import random
import numpy as np
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import uuid

# Step 1: CSV File Read
df = pd.read_csv('csv/아이돌.csv')

# Step 2: Model Setup
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs

def predict(inputs):
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = logits.softmax(dim=-1)
            return np.argmax(probs[0].detach().cpu().numpy())
    except Exception as e:
        print(f"Prediction error: {e}")
        return -1

df['evaluation'] = df['content'].apply(lambda x: predict(preprocess(x)))

def label_evaluation(index):
    if index == 0:
        return '부정'
    elif index == 1:
        return '긍정'
    else:
        return '알 수 없음'

df['result'] = df['evaluation'].apply(label_evaluation)

hannanum = Hannanum()
df['hannanum_tokens'] = df['content'].apply(hannanum.morphs)

# Step 3: Cassandra Connection Setup
auth_provider = PlainTextAuthProvider(username='cassandra', password='cassandra')
cluster = Cluster(['127.0.0.1'], auth_provider=auth_provider)
session = cluster.connect('advancedcassandra')  # Replace with your keyspace

# Step 4: Create Table if not exists
session.execute("""
CREATE TABLE IF NOT EXISTS news_sentiment_official (
    id UUID PRIMARY KEY,
    content text,
    result text,
    date text,
    link text
)
""")

# Step 5: Insert Data into Cassandra
for index, row in df.iterrows():
    doc = {
        'id': uuid.uuid4(),  # Generate a unique UUID for each record
        'content': row['content'],
        'result': row['result'],
        'date': row['date'],
        'link': row['link']
    }
    session.execute("""
    INSERT INTO news_sentiment_official (id, content, result, date, link) VALUES (%s, %s, %s, %s, %s)
    """, (doc['id'], doc['content'], doc['result'], doc['date'], doc['link']))

# Closing the connection
cluster.shutdown()

print("Data successfully inserted into Cassandra.")
