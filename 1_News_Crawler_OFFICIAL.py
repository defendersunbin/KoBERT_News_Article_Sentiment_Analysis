# 크롤링시 필요한 라이브러리 불러오기
from bs4 import BeautifulSoup
import requests
import re
import datetime
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob

# 페이지 url 형식에 맞게 바꿔 주는 함수
def makePgNum(num):
    if num == 1:
        return num
    elif num == 0:
        return num + 1
    else:
        return num + 9 * (num - 1)

# 크롤링할 url 생성하는 함수
def makeUrl(search, start_pg, end_pg):
    urls = []
    now = datetime.datetime.now()
    target_date = now - datetime.timedelta(days=7)
    target_date_str = target_date.strftime("%Y.%m.%d")

    for i in range(start_pg, end_pg + 1):
        page = makePgNum(i)
        url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={search}&sort=0&photo=0&field=0&pd=3&ds={target_date_str}&de={now.strftime('%Y.%m.%d')}&start={page}"
        #https://search.naver.com/search.naver?where=news&sm=tab_pge&query=뉴진스&sort=0&photo=0&field=0&pd=3&de=2024.11.26
        urls.append(url)

    print("생성 url: ", urls)
    return urls

# html에서 원하는 속성 추출하는 함수
def news_attrs_crawler(articles, attrs):
    attrs_content = []
    for i in articles:
        attrs_content.append(i.attrs[attrs])
    return attrs_content

# ConnectionError 방지
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}

# html 생성해서 기사 크롤링하는 함수
def articles_crawler(url):
    original_html = requests.get(url, headers=headers)
    html = BeautifulSoup(original_html.text, "html.parser")

    url_naver = html.select("div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
    return news_attrs_crawler(url_naver, 'href')

# 감성 분석 함수
def analyze_sentiment(content):
    analysis = TextBlob(content)
    return analysis.sentiment.polarity  # -1.0 ~ 1.0 사이의 점수

##### 뉴스 크롤링 시작 #####

# 검색어 입력
search = input("검색할 키워드를 입력해주세요: ")
# 검색 시작할 페이지 입력
page = int(input("\n크롤링할 시작 페이지를 입력해주세요. ex)1(숫자만 입력): "))
# 검색 종료할 페이지 입력
page2 = int(input("\n크롤링할 종료 페이지를 입력해주세요. ex)1(숫자만 입력): "))

# naver url 생성
url = makeUrl(search, page, page2)

# 뉴스 크롤러 실행
news_titles = []
news_url = []
news_contents = []
news_dates = []

for i in url:
    article_urls = articles_crawler(i)
    news_url.extend(article_urls)

# NAVER 뉴스만 남기기
final_urls = [url for url in news_url if "news.naver.com" in url]

# 뉴스 내용 크롤링
for i in tqdm(final_urls):
    news = requests.get(i, headers=headers)
    news_html = BeautifulSoup(news.text, "html.parser")

    # 뉴스 제목 가져오기
    title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
    title_text = title.get_text(strip=True) if title else "제목 없음"

    # 뉴스 본문 가져오기
    content_div = news_html.select_one("#dic_area")
    content_text = content_div.get_text(strip=True) if content_div else "내용 없음"

    # 감성 분석
    sentiment_score = analyze_sentiment(content_text)

    # 날짜 가져오기
    try:
        html_date = news_html.select_one("div#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")
        news_date = html_date.attrs['data-date-time']
    except AttributeError:
        news_date = "날짜 없음"

    # 결과 저장
    news_titles.append(title_text)
    news_contents.append(content_text)
    news_dates.append(news_date)

print("검색된 기사 갯수: 총 ", len(final_urls), '개')
print("\n[뉴스 제목]")
print(news_titles)
print("\n[뉴스 링크]")
print(final_urls)
print("\n[뉴스 내용]")
print(news_contents)

# 데이터 프레임 만들기
news_df = pd.DataFrame({
    'date': news_dates,
    'title': news_titles,
    'link': final_urls,
    'content': news_contents,
    'sentiment': [analyze_sentiment(content) for content in news_contents]
})

# 중복 행 지우기
news_df = news_df.drop_duplicates(keep='first', ignore_index=True)
print("중복 제거 후 행 개수: ", len(news_df))

# 데이터 프레임 저장
now = datetime.datetime.now()
news_df = news_df.sort_values(by='date', ascending=False)
news_df.to_csv('csv/뉴스_크롤링.csv', encoding='utf-8-sig', index=False)

print("크롤링 완료! 결과를 csv 파일로 저장했습니다.")
