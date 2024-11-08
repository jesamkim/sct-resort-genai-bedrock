import streamlit as st
import os
from PIL import Image
import pandas as pd
from googleapiclient.discovery import build
from anthropic import AnthropicBedrock
import random
from textblob import TextBlob
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
from io import BytesIO

BEDROCK_REGION = "us-east-1"  ## Claude 3 모델 호출 리전
YOUTUBE_API_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXX'   ## 자신의 YouTube Data v3 API Key로 변경


# URL에서 video ID를 추출하는 함수
def extract_video_id(url):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# 기존의 분석 관련 함수들
def get_replies(youtube, parent_id, video_id):
    replies = []
    request = youtube.comments().list(
        part="snippet",
        parentId=parent_id,
        textFormat="plainText",
        maxResults=100
    )
    while request:
        response = request.execute()
        replies.extend([{
            'Timestamp': item['snippet']['publishedAt'],
            'Username': item['snippet']['authorDisplayName'],
            'VideoID': video_id,
            'topLevelComment': response['items'][0]['snippet']['textDisplay'],
            'ThreadComment': item['snippet']['textDisplay'],
            'likeCount': item['snippet']['likeCount']
        } for item in response['items']])
        request = youtube.comments().list_next(request, response)
    return replies

def get_comments_for_video(youtube, video_id):
    all_comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100
    )
    while request:
        response = request.execute()
        for item in response['items']:
            top_comment = item['snippet']['topLevelComment']['snippet']
            all_comments.append({
                'Timestamp': top_comment['publishedAt'],
                'Username': top_comment['authorDisplayName'],
                'VideoID': video_id,
                'topLevelComment': top_comment['textDisplay'],
                'ThreadComment': '',
                'likeCount': top_comment['likeCount']
            })
            if item['snippet']['totalReplyCount'] > 0:
                all_comments.extend(get_replies(youtube, item['id'], video_id))
        request = youtube.commentThreads().list_next(request, response)
    return all_comments

def get_video_details(youtube, video_id):
    request = youtube.videos().list(
        part='snippet',
        id=video_id
    )
    response = request.execute()
    if response['items']:
        snippet = response['items'][0]['snippet']
        return {
            'VideoID': video_id,
            'Title': snippet['title'],
            'Description': snippet['description']
        }
    return None

def analyze_video_content_small(video_id, video_df, comments_df, max_chunk_length=5000):
    # 비디오 정보 가져오기
    video_info = video_df[video_df['VideoID'] == video_id].iloc[0]
    video_title = video_info['Title']
    video_description = video_info['Description']

    # 해당 비디오의 댓글 필터링
    video_comments = comments_df[comments_df['VideoID'] == video_id]

    # 모든 텍스트 결합
    all_text = f"Comments:\n"
    
    for _, comment in video_comments.iterrows():
        all_text += f"Top Level Comment: {comment['topLevelComment']}\n"
        if comment['ThreadComment']:
            all_text += f"Thread Comment: {comment['ThreadComment']}\n"
        all_text += "\n"

    # 텍스트 청크로 나누기
    # chunks = chunk_text(all_text, max_length=max_chunk_length)

    all_analyses = []
    # for chunk in chunks:
    prompt = f"""
    비디오 제목: {video_info['Title']}
    비디오 설명: {video_info['Description']}
    
        다음 YouTube 비디오 콘텐츠(제목, 설명, Top Level Comment, Thread Comment 포함)를 분석해주세요. 당신의 임무는 다음과 같습니다:

        1. 제목, 설명, Top Level Comment, Thread Comment, 좋야요 수를 바탕으로 비디오에서 논의된 주요 주제나 테마를 요약해주세요.
        2. Thread Comment 분석시 Top Level Comment의 내용과 Thread Comment의 Timestamp를 보고 Context를 같이 분석해주세요.
        3. 댓글의 전반적인 감정(긍정적, 부정적, 중립적)을 Context에 맞게 파악해주고 각 감정별로 원문을 정리해주세요.
        4. Thread Comment가 많이 있는 Top Level Comment의 댓글을 분석하고 주된 논의 내용 및 반복적인 키워드를 나열해주고 대표적인 내용의 원문을 보여주세요.
        5. 댓글 작성자들이 반복적으로 제기하는 질문, 우려사항, 관심사를 강조해주고 원문을 보여주세요.
        6. 댓글에서 나타나는 중요한 의견 차이나 논쟁을 파악해주고(좋아요 수를 참고해주세요) 원문을 보여주세요. 
        7. 비디오 제작자에게 제시된 주목할 만한 피드백이나 제안을 확인해주고 원문을 보여주세요.

        비디오 설명이 제공하는 컨텍스트와 댓글 작성자들 간의 상호작용을 고려하여 종합적인 분석을 제공해주세요. 모든 응답은 한국어로 작성해 주세요.
        분석에 근거가 되는 원문을 포함해서 작성해주세요.
        Video Content:
        {all_text}
        """
    print(f"small prompt: {len(prompt)}")
    print(f"small prompt: {len(prompt.encode('utf-8'))}")

    response = client.messages.create(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            max_tokens=4096,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    print(f"small 3.5 sonnet Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")


    result = response.content[0].text
    all_analyses.append(result)

    return " ".join(all_analyses)

def get_top_comments(comments, n=100):
    return sorted(comments, key=lambda x: x['likeCount'], reverse=True)[:n]

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def cluster_comments(comments, n_clusters=5):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform([c['topLevelComment'] for c in comments])
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.labels_

def analyze_video_content_large(video_id, video_df, comments_df, max_chunk_length=5000):
    video_info = video_df[video_df['VideoID'] == video_id].iloc[0]
    video_comments = comments_df[comments_df['VideoID'] == video_id]
    
    # 인기 댓글 분석
    top_comments = get_top_comments(video_comments.to_dict('records'), n=100)
    
    # 감정 분석
    sentiments = [analyze_sentiment(c['topLevelComment']) for c in top_comments]
    avg_sentiment = sum(sentiments) / len(sentiments)
    
    # 클러스터링
    cluster_labels = cluster_comments(video_comments.to_dict('records'))
    
    # Thread가 많은 댓글 필터링
    comments_with_threads = [c for c in video_comments.to_dict('records') if c['ThreadComment']]
    top_thread_comments = sorted(comments_with_threads, key=lambda x: len(x['ThreadComment']), reverse=True)[:10]
    
    
    # Thread가 많은 댓글 그룹핑 및 분석
    thread_comment_analysis = ""
    input_tokens_result =0
    output_tokens_result =0
    for c in top_thread_comments:
        top_comment = c['topLevelComment']
        thread_comments = c['ThreadComment']
        
        # 각 그룹에 대해 감정 분석, 주요 주제 및 키워드 분석
        all_comments_text = f"Top Level Comment: {top_comment}\n" + "\n".join(thread_comments)
        thread_sentiment = analyze_sentiment(all_comments_text)
        
        # 대표적인 댓글과 주요 주제 추출
        prompt= f"""Top Level Comment와 Thread Comments를 포함한 다음 댓글 그룹을 분석해주세요:Top Level Comment: {top_comment} 
        Thread Comments: {''.join([f'- {t}, ' for t in thread_comments])} 
        
        1. 분석에 근거가 되는 원문을 포함해서 이 그룹에서 논의된 주요 주제나 테마를 요약해주세요.
        2. 분석에 근거가 되는 원문을 포함해서 전체적인 감정(긍정적, 부정적, 중립적)을 파악해주세요.
        3. 분석에 근거가 되는 원문을 포함해서 반복적으로 나타나는 키워드나 표현을 요약해주세요.
        4. 분석에 근거가 되는 원문을 포함해서 중요한 논쟁이나 의견 차이가 있는 경우 이를 식별하고 설명해주세요.
        5. 분석에 근거가 되는 원문을 포함해서 주요 논의 내용 및 전반적인 톤을 요약해주세요.
        6. 분석근거의 원문을 생략없이 최대한 많이 포함해서 보여주세요.
        """
        
        print(f"haiku len prompt: {len(prompt)}")
        print(f"haiku encode utf8 len prompt: {len(prompt.encode('utf-8'))}")
        
        response = client.messages.create(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            max_tokens=1500,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        print(f"haiku each Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")

        input_tokens_result += input_tokens
        output_tokens_result += output_tokens
        
        thread_analysis = response.content[0].text
        
        thread_comment_analysis += f"\nTop Level Comment: {top_comment[:50]}...\n"
        thread_comment_analysis += f"Sentiment: {thread_sentiment}\n"
        thread_comment_analysis += f"Analysis:\n{thread_analysis}\n\n"
        
    print(f"haiku input token: {input_tokens_result}")
    print(f"haiku output token: {output_tokens_result}")
        
    
    # 분석 결과 요약
    summary = f"""비디오 제목: {video_info['Title']}
    비디오 설명: {video_info['Description']}
       
    1. 상위 인기 댓글 (좋아요 순 상위 30개):
    {
        ''.join([f"   - {c['topLevelComment'][:20]}... (좋아요: {c['likeCount']}... {c['ThreadComment'][:200]}) " for c in top_comments[:30]])
    }
    
    2. 주요 댓글 클러스터:
    {
        ''.join([f"   클러스터 {i}: {Counter(cluster_labels)[i]}개 댓글 " for i in range(100)])
    }
    
    3. Thread가 많은 상위 10개 댓글 분석:
    {thread_comment_analysis}
    
    4. 전반적인 분석:
      1) 분석에 근거가 되는 원문을 포함해서 댓글의 전반적인 감정(긍정적, 부정적, 중립적)을 파악해주세요.
      2) 분석에 근거가 되는 원문을 포함해서 토론의 주된 톤(예: 열정적, 비판적, 정보 제공적, 유머러스 등)을 결정해주세요.
      3) 분석에 근거가 되는 원문을 포함해서 댓글 작성자들이 반복적으로 제기하는 질문, 우려사항, 관심사를 강조해주세요.
      4) 분석에 근거가 되는 원문을 포함해서 댓글에서 나타나는 중요한 의견 차이나 논쟁을 파악해주세요.
      5) 분석에 근거가 되는 원문을 포함해서 비디오 제작자에게 제시된 주목할 만한 피드백이나 제안을 확인해주세요.
      6) 분석에 근거가 되는 원문을 포함해서 Thread가 많은 상위 10개 댓글 요약 분석에 대한 내용을 포함해서 별도로 작성해주고 안에 원문은 그대로 유지하세요.

        비디오 설명이 제공하는 컨텍스트와 댓글 작성자들 간의 상호작용을 고려하여 종합적인 분석을 제공해주세요.
        분석에 근거가 되는 원문을 생략없이 최대한 많이 포함해서 작성해주세요.
    """
    
    print(f"large summary prompt: {len(prompt)}")
    print(f"large summary prompt: {len(prompt.encode('utf-8'))}")
    
    response = client.messages.create(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            max_tokens=4096,
            temperature=0.7,
            messages=[
                {"role": "user", "content": summary}
            ]
        )
    
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    print(f"large 3.5 sonnet input token: {input_tokens}")
    print(f"large 3.5 sonnet output token: {output_tokens}")

    result = response.content[0].text

    return result
    
# Function to get replies for a specific comment
def get_replies(youtube, parent_id, video_id):
    replies = []
    request = youtube.comments().list(
        part="snippet",
        parentId=parent_id,
        textFormat="plainText",
        maxResults=100
    )
    
    while request:
        response = request.execute()
        replies.extend([{
            'Timestamp': item['snippet']['publishedAt'],
            'Username': item['snippet']['authorDisplayName'],
            'VideoID': video_id,
            'topLevelComment': response['items'][0]['snippet']['textDisplay'],  # Avoid redundant indexing
            'ThreadComment': item['snippet']['textDisplay'],
            'likeCount': item['snippet']['likeCount']
            # 'Date': item['snippet'].get('updatedAt', item['snippet']['publishedAt'])
        } for item in response['items']])

        request = youtube.comments().list_next(request, response)

    return replies


# Function to get all comments (including replies) for a single video
def get_comments_for_video(youtube, video_id):
    all_comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100
    )
    
    while request:
        response = request.execute()
        for item in response['items']:
            top_comment = item['snippet']['topLevelComment']['snippet']
            all_comments.append({
                'Timestamp': top_comment['publishedAt'],
                'Username': top_comment['authorDisplayName'],
                'VideoID': video_id,
                'topLevelComment': top_comment['textDisplay'],
                'ThreadComment': '',
                'likeCount': top_comment['likeCount']
                # 'Date': top_comment.get('updatedAt', top_comment['publishedAt'])
            })

            # Fetch replies if there are any
            if item['snippet']['totalReplyCount'] > 0:
                all_comments.extend(get_replies(youtube, item['id'], video_id))

        request = youtube.commentThreads().list_next(request, response)

    return all_comments


# Function to get video details (title and description)
def get_video_details(youtube, video_id):
    request = youtube.videos().list(
        part='snippet',
        id=video_id
    )
    response = request.execute()

    if response['items']:
        snippet = response['items'][0]['snippet']
        return {
            'VideoID': video_id,
            'Title': snippet['title'],
            'Description': snippet['description']
        }
    return None



# Streamlit 페이지 설정
st.set_page_config(layout="wide", page_title="SCT Resort GenAI #2")
logo = Image.open('logo.png')
st.image(logo, width=200)
st.title(":blue[Samsung C&T Resort - Analyzing YouTube comments]")
st.caption(":rainbow[powered by Amazon Bedrock]") 


if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# 화면 초기화 버튼 추가
if st.button("화면 초기화"):
    st.session_state.analysis_results = None
    st.rerun()

# 입력 UI 구성
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.write("YouTube URL 입력")
with col2:
    youtube_url = st.text_input("", label_visibility="collapsed")
with col3:
    analyze_button = st.button("댓글 분석", type="primary", use_container_width=True)

# 분석 실행
if analyze_button and youtube_url:
    try:
        # YouTube API 초기화
        api_service_name = "youtube"
        api_version = "v3"
        youtubeAPI_key = YOUTUBE_API_KEY
        youtube = build(api_service_name, api_version, developerKey=youtubeAPI_key)

        # AnthropicBedrock 클라이언트 초기화
        client = AnthropicBedrock(aws_region=BEDROCK_REGION)  ## Bedrock 리전

        # video ID 추출
        video_id = extract_video_id(youtube_url)
        if video_id:
            # 결과를 담을 컨테이너 생성
            with st.container():
                # 로딩 표시
                with st.spinner('댓글을 분석중입니다...'):
                    # 비디오 정보와 댓글 수집
                    video_details = []
                    all_comments = []
                    video_info = get_video_details(youtube, video_id)
                    if video_info:
                        video_details.append(video_info)
                        video_comments = get_comments_for_video(youtube, video_id)
                        all_comments.extend(video_comments)

                    video_df = pd.DataFrame(video_details)
                    comments_df = pd.DataFrame(all_comments)

                    # 분석 실행
                    analysis_results = []
                    for _, row in video_df.iterrows():
                        vid = row['VideoID']
                        title = row['Title']
                        if len(comments_df[comments_df['VideoID'] == vid]) > 1000:
                            analysis_type = 'large'
                            analysis = analyze_video_content_large(vid, video_df, comments_df)
                        else:
                            analysis_type = 'small'
                            analysis = analyze_video_content_small(vid, video_df, comments_df)
                        
                        analysis_results.append({
                            'VideoID': vid,
                            'Title': title,
                            'Analysis-type': analysis_type,
                            'Analysis': analysis
                        })

                    analysis_df = pd.DataFrame(analysis_results)

                    # 분석 결과를 session_state에 저장
                    st.session_state.analysis_results = {
                        'youtube_url': youtube_url,
                        'comments_df': comments_df,
                        'analysis_df': analysis_df
                    }

        else:
            st.error("올바른 YouTube URL을 입력해주세요.")
    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {str(e)}")

# 저장된 결과가 있으면 표시
if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    
    # 결과 출력
    st.markdown("### 게시물의 분석결과는 다음과 같습니다.")

    # 테두리가 있는 박스 스타일 정의
    st.markdown("""
    <style>
    .result-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

    # 테두리가 있는 박스에 결과 표시
    st.markdown(f"""
    <div class="result-box">
    • 게시물 URL: {results['youtube_url']}<br>
    • 총 댓글수: {len(results['comments_df'][results['comments_df']['ThreadComment'] == ''])}건<br>
    • 대댓글수: {len(results['comments_df'][results['comments_df']['ThreadComment'] != ''])}건
    </div>
    """, unsafe_allow_html=True)

    # 상세 분석 결과
    st.markdown("### 상세 분석 결과")
    for _, row in results['analysis_df'].iterrows():
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 10px 0;'>
            <p><strong>Video ID:</strong><br>{}</p>
            <p><strong>Title:</strong><br>{}</p>
            <p><strong>Analysis Type:</strong><br>{}</p>
            <p><strong>Analysis:</strong><br>{}</p>
        </div>
        """.format(
            row['VideoID'],
            row['Title'],
            row['Analysis-type'],
            row['Analysis'].replace('\n', '<br>')
        ), unsafe_allow_html=True)

    # 엑셀 파일 생성을 위한 함수
    def to_excel_bytes(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Analysis Results', index=False)
            # 자동 열 너비 조정
            worksheet = writer.sheets['Analysis Results']
            for idx, col in enumerate(df.columns):
                series = df[col]
                max_len = max(
                    series.astype(str).map(len).max(),  # 열의 최대 문자 길이
                    len(str(col))  # 열 이름의 길이
                ) + 2  # 여유 공간
                worksheet.set_column(idx, idx, max_len)
        
        output.seek(0)
        return output.getvalue()

    st.markdown("---")
    
    # 엑셀 파일 생성
    excel_data = to_excel_bytes(results['analysis_df'])
    
    # video_id 추출
    video_id = extract_video_id(results['youtube_url'])
    
    # 다운로드 버튼
    st.download_button(
        label="Excel 결과 다운로드",
        data=excel_data,
        file_name=f'youtube_analysis_result_{video_id}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
