from cassandra.cluster import Cluster
import pandas as pd
import uuid

# Cassandra 클러스터에 연결
cluster = Cluster(['127.0.0.1'])  # 호스트를 적절하게 변경하세요
session = cluster.connect()

# Step 1: 기존 테이블에서 데이터 불러오기
rows = session.execute("SELECT content, result FROM advancedcassandra.news_sentiment")

# 데이터프레임으로 변환
df = pd.DataFrame(rows, columns=['content', 'result'])

# Step 2: 새로운 테이블 생성
session.execute("""
CREATE TABLE IF NOT EXISTS advancedcassandra.news_sentiment_official (
    id UUID PRIMARY KEY,
    content text,
    result text
)
""")

# Step 3: 데이터 삽입
for index, row in df.iterrows():
    doc = {
        'id': uuid.uuid4(),  # 각 레코드에 대해 고유한 UUID 생성
        'content': row['content'],
        'result': row['result']
    }
    session.execute("""
    INSERT INTO advancedcassandra.news_sentiment_official (id, content, result) VALUES (%s, %s, %s)
    """, (doc['id'], doc['content'], doc['result']))

# Step 4: 연결 종료
cluster.shutdown()

print("Data successfully inserted into news_sentiment_official.")
