# KoBERT_NewsTitleEmotionAnalysis
 
KoBert 모델을 이용하여 뉴스기사 내용 전체를 감성분석하는 시스템 및 웹사이트입니다.

먼저 Naver News Search API를 이용해서 뉴스기사를 크롤링합니다.

그 후 DB에 필요한 내용들을 얻습니다. - 제목, 날짜(시간), 라벨링 결과(float), 감성분석 결과(Positive, Negative)

현재는 긍정, 부정만을 위주로 감성분석이 가능합니다. 추후 중립에 관한 감성분석 사전도 따로 만들 예정입니다.

KoBERT 감성분석 모델을 만들어주신 개발자님께 감사의 말씀을 드립니다.

해당 작성자는 Cassandra DataBase를 구축하여 크롤링한 뉴스기사 내용 및 필수 요소들을 Cassandra DataBase Table에 Insert, Create Table SQL 쿼리를 추가함으로써 데이터를 삽입할 수 있도록 했습니다.

마지막으로 Flask + Rest API를 구축해 웹사이트를 만들어서 실제로 키워드를 검색하면 해당 뉴스기사 내용에 따라 어떤 감성분석 결과를 얻을 수 있는지 시각적으로 볼 수 있습니다.
